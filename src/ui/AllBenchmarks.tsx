import React, { useEffect, useState } from 'react';
import { Benchmark } from './Benchmark';
import { bigIntToU32Array, generateRandomFields } from '../reference/webgpu/utils';
import { BigIntPoint, U32ArrayPoint } from '../reference/types';
import { webgpu_compute_msm, wasm_compute_msm, webgpu_pippenger_msm, webgpu_best_msm, wasm_compute_msm_parallel } from '../reference/reference';
import { convert_inputs_into_mont_benchmark } from '../submission/implementation/convert_inputs_into_mont_benchmarks';
import { convert_bigints_to_bytes_benchmark } from '../submission/implementation/convert_bigints_to_bytes_benchmark'
import { mont_mul_benchmarks } from '../submission/implementation/mont_mul_benchmarks';
import { barrett_mul_benchmarks } from '../submission/implementation/barrett_mul_benchmarks';
import { barrett_domb_mul_benchmarks } from '../submission/implementation/barrett_domb_mul_benchmarks';
import { add_points_benchmarks } from '../submission/implementation/add_points_benchmarks';
import { decompose_scalars_ts_benchmark } from '../submission/implementation/decompose_scalars_benchmark';
import {
    create_csr_precomputation_benchmark,
    create_csr_sparse_matrices_from_points_benchmark,
} from '../submission/implementation/cuzk/create_csr_gpu'
import { full_benchmarks } from '../submission/implementation/full_benchmarks'
import { scalar_mul_benchmarks } from '../submission/implementation/scalar_mul_benchmarks'
import { smtvp_wgsl } from '../submission/implementation/cuzk/smtvp_wgsl';
import { cuzk_typescript_serial } from '../submission/implementation/cuzk/cuzk_serial'
import { transpose_wgsl } from '../submission/implementation/cuzk/transpose_wgsl'
import CSVExportButton from './CSVExportButton';
import { TestCaseDropDown } from './TestCaseDropDown';
import { PowersTestCase, TestCase, loadTestCase } from '../test-data/testCases';
import { smvp_wgsl } from '../submission/implementation/cuzk/smvp_wgsl';
import { data_transfer_cost_benchmarks } from '../submission/implementation/data_transfer_cost_benchmarks'
import { bucket_points_reduction } from '../submission/implementation/bucket_points_reduction_benchmark'
import { horners_rule_benchmark } from '../submission/implementation/horners_rule_benchmark'
import { print_device_limits } from '../submission/implementation/print_device_limits'
import { compute_msm } from '../submission/submission';

export const AllBenchmarks: React.FC = () => {
  const initialDefaultInputSize = 2 ** 16
  const [inputSize, setInputSize] = useState(initialDefaultInputSize);
  const [power, setPower] = useState<string>('2^0');
  const [inputSizeDisabled, setInputSizeDisabled] = useState(false);
  const [baseAffineBigIntPoints, setBaseAffineBigIntPoints] = useState<BigIntPoint[]>([]);
  const [bigIntScalars, setBigIntScalars] = useState<bigint[]>([]);
  const [u32Points, setU32Points] = useState<U32ArrayPoint[]>([]);
  const [u32Scalars, setU32Scalars] = useState<Uint32Array[]>([]);
  const [expectedResult, setExpectedResult] = useState<{x: bigint, y: bigint} | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [benchmarkResults, setBenchmarkResults] = useState<any[][]>([["InputSize", "MSM Func", "Time (MS)"]]);
  const [comparisonResults, setComparisonResults] = useState<{ x: bigint, y: bigint, timeMS: number, msmFunc: string, inputSize: number }[]>([]);
  const [disabledBenchmark, setDisabledBenchmark] = useState<boolean>(false);

  const postResult = (result: {x: bigint, y: bigint}, timeMS: number, msmFunc: string) => {
    const benchMarkResult = [inputSizeDisabled ? power : inputSize, msmFunc, timeMS];
    setBenchmarkResults([...benchmarkResults, benchMarkResult]);
    setComparisonResults([...comparisonResults, {x: result.x, y: result.y, timeMS, msmFunc, inputSize}]);
    if (msmFunc.includes('Aleo Wasm')) {
      setExpectedResult(result);
    }
  };

  const loadAndSetData = async (power: PowersTestCase) => {
    setInputSizeDisabled(true);
    setInputSize(0);
    setDisabledBenchmark(true);
    setPower(`2^${power}`);
    const testCase = await loadTestCase(power);
    setTestCaseData(testCase);
  }

  const setTestCaseData = async (testCase: TestCase) => {
    setBaseAffineBigIntPoints(testCase.baseAffinePoints);
    const newU32Points = testCase.baseAffinePoints.map((point) => {
      return {
        x: bigIntToU32Array(point.x),
        y: bigIntToU32Array(point.y),
        t: bigIntToU32Array(point.t),
        z: bigIntToU32Array(point.z),
      }});
    setU32Points(newU32Points);
    setBigIntScalars(testCase.scalars);
    const newU32Scalars = testCase.scalars.map((scalar) => bigIntToU32Array(scalar));
    setU32Scalars(newU32Scalars);
    setExpectedResult(testCase.expectedResult);
    setDisabledBenchmark(false);
  };

  const useRandomInputs = () => {
    setDisabledBenchmark(true);
    setInputSizeDisabled(false);
    setExpectedResult(null);
    setInputSize(initialDefaultInputSize);
    setDisabledBenchmark(false);
  };

  useEffect(() => {
    async function generateNewInputs() {
      // creating random points is slow, so for now use a single fixed base.
      // const newPoints = await createRandomAffinePoints(inputSize);
      const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246');
      const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166');
      const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023');
      const z = BigInt('1');
      const point: BigIntPoint = {x, y, t, z};
      const newPoints = Array(inputSize).fill(point);
      setBaseAffineBigIntPoints(newPoints);
      const newU32Points = newPoints.map((point) => {
        return {
          x: bigIntToU32Array(point.x),
          y: bigIntToU32Array(point.y),
          t: bigIntToU32Array(point.t),
          z: bigIntToU32Array(point.z),
        }});
      setU32Points(newU32Points);

      //const newScalars = generateRandomFields(inputSize);
    
      // Use constants instead of random field elements just for testing
      const newScalars: bigint[] = []
      for (let i = 0; i < inputSize; i ++) {
          const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
          const x = BigInt('1111111111111111111111111111111111111111111111111111111111111111111111111111')
          newScalars.push((x * BigInt(i) % p))
      }
    /*
     */
        
      setBigIntScalars(newScalars);
      const newU32Scalars = newScalars.map((scalar) => bigIntToU32Array(scalar));
      setU32Scalars(newU32Scalars);
    }
    generateNewInputs();
    setComparisonResults([]);
  }, [inputSize]);
  
  return (
    <div>
      <div className="flex items-center space-x-4 px-5">
        <div className="text-gray-800">Input Size:</div>
        <input
          type="text"
          className="w-24 border border-gray-300 rounded-md px-2 py-1"
          value={inputSize}
          disabled={inputSizeDisabled}
          onChange={(e) => setInputSize(parseInt(e.target.value))}
        />
        <TestCaseDropDown useRandomInputs={useRandomInputs} loadAndSetData={loadAndSetData}/>
        <CSVExportButton data={benchmarkResults} filename={'msm-benchmark'} />
      </div>
      <Benchmark
        name={'Submission'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={compute_msm}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Full benchmark suite'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={full_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Horner\s Rule'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={horners_rule_benchmark}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Print device limits'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={print_device_limits}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Bucket points reduction'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={bucket_points_reduction}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Scalar multiplication benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={scalar_mul_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Data transfer cost benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={data_transfer_cost_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Pippenger WebGPU MSM'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={webgpu_pippenger_msm}
        postResult={postResult}
      />
      <Benchmark
        name={'Naive WebGPU MSM'}
        disabled={disabledBenchmark}
        // baseAffinePoints={u32Points}
        // scalars={u32Scalars}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={webgpu_compute_msm}
        postResult={postResult}
      />
      <Benchmark
        name={'Aleo Wasm: Single Thread'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={wasm_compute_msm}
        postResult={postResult}
      />
      <Benchmark
        name={'Aleo Wasm: Web Workers'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={wasm_compute_msm_parallel}
        postResult={postResult}
      />
      <Benchmark
        name={'Our Best WebGPU MSM'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={webgpu_best_msm}
        postResult={postResult}
        bold={false}
      />

      <Benchmark
        name={'Decompose scalars benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={decompose_scalars_ts_benchmark}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Montgomery multiplication benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={mont_mul_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Barrett reduction benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={barrett_mul_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Barrett-Domb reduction benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={barrett_domb_mul_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Convert point coordinates to Mont form benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={convert_inputs_into_mont_benchmark}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'BigInts to bytes benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={convert_bigints_to_bytes_benchmark}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'cuzk Serial (Typescript)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={cuzk_typescript_serial}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Point addition algorithm benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={add_points_benchmarks}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Transpose (WGSL) - classic algo'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={transpose_wgsl}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'SMVP (WGSL)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={smvp_wgsl}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'SMTVP (WGSL)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={smtvp_wgsl}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Create CSR sparse matrices (precomputation only in TS)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={create_csr_precomputation_benchmark}
        postResult={postResult}
        bold={true}
      />
      <Benchmark
        name={'Create CSR sparse matrices (GPU)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={create_csr_sparse_matrices_from_points_benchmark}
        postResult={postResult}
        bold={true}
      />
    </div>
  )
};
