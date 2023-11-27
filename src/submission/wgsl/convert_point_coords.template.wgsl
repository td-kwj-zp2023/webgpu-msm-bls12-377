{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> bigint_funcs }}
{{> barrett_functions }}

// Input buffers
@group(0) @binding(0)
var<storage, read> x_y_coords: array<BigInt>;

// Output buffers
@group(0) @binding(1)
var<storage, read_write> point_x_y: array<BigInt>;
@group(0) @binding(2)
var<storage, read_write> point_t_z: array<BigInt>;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    var a: BigInt = x_y_coords[gidx * 2u];
    var b: BigInt = x_y_coords[(gidx * 2u) + 1u];
    var r = get_r();
    
    // Convert x and y coordinates to Montgomery form
    var xr = field_mul(&a, &r);
    var yr = field_mul(&b, &r);
    point_x_y[gidx * 2u] = xr;
    point_x_y[(gidx * 2u) + 1u] = yr;

    // Compute t
    let tr = montgomery_product(&xr, &yr);
    point_t_z[gidx * 2u] = tr;

    // Store z
    point_t_z[(gidx * 2u) + 1u] = r;
}
