import { Curve, base_field_modulus } from "./bls12_377";
import mustache from "mustache";
import convert_point_coords_and_decompose_scalars from "../wgsl/cuzk/convert_point_coords_and_decompose_scalars.template.wgsl";
import extract_word_from_bytes_le_funcs from "../wgsl/cuzk/extract_word_from_bytes_le.template.wgsl";
import structs from "../wgsl/struct/structs.template.wgsl";
import bigint_funcs from "../wgsl/bigint/bigint.template.wgsl";
import field_funcs from "../wgsl/field/field.template.wgsl";
import ec_bls12_377_funcs from "../wgsl/curve/ec_bls12_377.template.wgsl";
import barrett_funcs from "../wgsl/cuzk/barrett.template.wgsl";
import montgomery_product_funcs from "../wgsl/montgomery/mont_pro_product.template.wgsl";
import transpose_serial_shader from "../wgsl/cuzk/transpose_serial.wgsl";
import smvp_bls12_377_shader from "../wgsl/cuzk/smvp_bls12_377.template.wgsl";
import bpr_shader from "../wgsl/cuzk/bpr.template.wgsl";
import {
  compute_misc_params,
  gen_p_limbs,
  gen_r_limbs,
  gen_mu_limbs,
} from "./utils";

// A helper class which allows cuzk_gpu() to generate all the shaders it needs
// easily. It precomputes all the necessary variables (such as the Montgomery
// radix) which depend on the word size.
export class ShaderManager {
  public p: bigint;
  public word_size: number;
  public chunk_size: number;
  public input_size: number;
  public num_words: number;
  public index_shift: number;
  public mask: number;
  public two_pow_word_size: number;
  public two_pow_chunk_size: number;
  public n0: bigint;
  public r: bigint;
  public rinv: bigint;
  public p_bitlength: number;
  public slack: number;
  public w_mask: number;
  public p_limbs: string;
  public r_limbs: string;
  public mu_limbs: string;
  public recompile = "";

  constructor(
    word_size: number,
    chunk_size: number,
    input_size: number,
    force_recompile = false,
  ) {
    this.p = base_field_modulus[Curve.BLS12_377];
    const params = compute_misc_params(this.p, word_size);
    this.word_size = word_size;
    this.chunk_size = chunk_size;
    this.input_size = input_size;
    this.n0 = params.n0;
    this.num_words = params.num_words;
    this.r = params.r;
    this.rinv = params.rinv;
    this.mask = 2 ** word_size - 1;
    this.index_shift = 2 ** (chunk_size - 1);
    this.two_pow_word_size = 2 ** word_size;
    this.two_pow_chunk_size = 2 ** chunk_size;
    this.p_limbs = gen_p_limbs(this.p, this.num_words, word_size);
    this.r_limbs = gen_r_limbs(this.r, this.num_words, word_size);
    this.mu_limbs = gen_mu_limbs(this.p, this.num_words, word_size);
    this.p_bitlength = this.p.toString(2).length;
    this.slack = this.num_words * word_size - this.p_bitlength;
    this.w_mask = (1 << word_size) - 1;

    if (force_recompile) {
      const rand = Math.round(Math.random() * 100000000000000000) % 2 ** 32;
      this.recompile = `
                var recompile = ${rand}u;
                recompile += 1u;
            `.trim();
    }
  }

  public gen_convert_points_and_decomp_scalars_shader(
    workgroup_size: number,
    num_y_workgroups: number,
    num_subtasks: number,
    num_columns: number,
  ): string {
    const num_16_bit_words_per_coord = Math.ceil(
      (this.num_words * this.word_size) / 16,
    );
    const shaderCode = mustache.render(
      convert_point_coords_and_decompose_scalars,
      {
        workgroup_size,
        num_y_workgroups,
        num_subtasks,
        num_columns,
        num_words: this.num_words,
        word_size: this.word_size,
        n0: this.n0,
        mask: this.mask,
        two_pow_word_size: this.two_pow_word_size,
        two_pow_chunk_size: this.two_pow_chunk_size,
        index_shift: this.index_shift,
        p_limbs: this.p_limbs,
        r_limbs: this.r_limbs,
        mu_limbs: this.mu_limbs,
        w_mask: this.w_mask,
        slack: this.slack,
        num_words_mul_two: this.num_words * 2,
        num_words_plus_one: this.num_words + 1,
        chunk_size: this.chunk_size,
        input_size: this.input_size,
        num_16_bit_words_per_coord,
        recompile: this.recompile,
      },
      {
        structs,
        bigint_funcs,
        field_funcs,
        barrett_funcs,
        montgomery_product_funcs,
        extract_word_from_bytes_le_funcs,
      },
    );
    return shaderCode;
  }

  public gen_transpose_shader(workgroup_size: number) {
    const shaderCode = mustache.render(
      transpose_serial_shader,
      {
        workgroup_size,
        recompile: this.recompile,
      },
      {},
    );
    return shaderCode;
  }

  public gen_smvp_shader(workgroup_size: number, num_csr_cols: number) {
    const shaderCode = mustache.render(
      smvp_bls12_377_shader,
      {
        word_size: this.word_size,
        num_words: this.num_words,
        n0: this.n0,
        p_limbs: this.p_limbs,
        r_limbs: this.r_limbs,
        mask: this.mask,
        two_pow_word_size: this.two_pow_word_size,
        index_shift: this.index_shift,
        workgroup_size,
        num_columns: num_csr_cols,
        half_num_columns: num_csr_cols / 2,
        recompile: this.recompile,
      },
      {
        structs,
        bigint_funcs,
        montgomery_product_funcs,
        field_funcs,
        ec_funcs: ec_bls12_377_funcs,
      },
    );
    return shaderCode;
  }

  public gen_bpr_shader(workgroup_size: number) {
    const shaderCode = mustache.render(
      bpr_shader,
      {
        word_size: this.word_size,
        num_words: this.num_words,
        n0: this.n0,
        p_limbs: this.p_limbs,
        r_limbs: this.r_limbs,
        mask: this.mask,
        two_pow_word_size: this.two_pow_word_size,
        index_shift: this.index_shift,
        workgroup_size,
        recompile: this.recompile,
      },
      {
        structs,
        bigint_funcs,
        montgomery_product_funcs,
        field_funcs,
        ec_funcs: ec_bls12_377_funcs,
      },
    );
    return shaderCode;
  }
}
