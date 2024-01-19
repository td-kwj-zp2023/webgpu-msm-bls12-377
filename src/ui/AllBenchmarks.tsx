import React, { useEffect, useState } from 'react';
import { Benchmark } from './Benchmark';
import { bigIntToU32Array } from '../reference/webgpu/utils';
import { BigIntPoint, U32ArrayPoint } from '../reference/types';
import { webgpu_compute_msm, wasm_compute_msm, webgpu_pippenger_msm, webgpu_best_msm, wasm_compute_msm_parallel } from '../reference/reference';
import { convert_inputs_into_mont_benchmark } from '../submission/convert_inputs_into_mont_benchmarks';
import { convert_bigints_to_bytes_benchmark } from '../submission/convert_bigints_to_bytes_benchmark'
import { mont_mul_benchmarks } from '../submission/mont_mul_benchmarks';
import { barrett_mul_benchmarks } from '../submission/barrett_mul_benchmarks';
import { barrett_domb_mul_benchmarks } from '../submission/barrett_domb_mul_benchmarks';
import { add_points_benchmarks } from '../submission/add_points_benchmarks';
import { decompose_scalars_ts_benchmark } from '../submission/decompose_scalars_benchmark';
import {
    create_csr_precomputation_benchmark,
    create_csr_sparse_matrices_from_points_benchmark,
} from '../submission/cuzk/create_csr_gpu'
import { cuzk_gpu } from '../submission/cuzk/cuzk_gpu'
import { full_benchmarks } from '../submission/full_benchmarks'
import { scalar_mul_benchmarks } from '../submission/scalar_mul_benchmarks'
import { cuzk_typescript_serial, transpose_wgsl, smtvp_wgsl, smvp_wgsl } from '../submission/submission';
import CSVExportButton from './CSVExportButton';
import { TestCaseDropDown } from './TestCaseDropDown';
import { PowersTestCase, TestCase, loadTestCase } from '../test-data/testCases';
import { data_transfer_cost_benchmarks } from '../submission/data_transfer_cost_benchmarks'
import { bucket_points_reduction } from '../submission/bucket_points_reduction_benchmark'
import { horners_rule_benchmark } from '../submission/horners_rule_benchmark'
import { print_device_limits } from '../submission/print_device_limits'

export const AllBenchmarks: React.FC = () => {
  const initialDefaultInputSize = 2 ** 16
  const [inputSize, setInputSize] = useState(initialDefaultInputSize);
  const [power, setPower] = useState<string>('2^0');
  const [inputSizeDisabled, setInputSizeDisabled] = useState(false);
  const [baseAffineBigIntPoints, setBaseAffineBigIntPoints] = useState<BigIntPoint[]>([]);
  const [bigIntScalars, setBigIntScalars] = useState<bigint[]>([]);
  const [{}, setU32Points] = useState<U32ArrayPoint[]>([]);
  const [{}, setU32Scalars] = useState<Uint32Array[]>([]);
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
        name={'cuZK in WebGPU'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={cuzk_gpu}
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
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={webgpu_compute_msm}
        postResult={postResult}
      />
      <Benchmark
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

      <div className="flex items-left">
        <div className={`text-gray-800 w-40 px-2 font-bold'`}>{name}</div> 
          <h1>Miscellaneous benchmarks for testing</h1>
          <br />
          <br />
      </div>

      <Benchmark
        name={'Full benchmark suite'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={full_benchmarks}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Horner\'s Rule'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={horners_rule_benchmark}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Bucket points reduction'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={bucket_points_reduction}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Scalar multiplication benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={scalar_mul_benchmarks}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Data transfer cost benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={data_transfer_cost_benchmarks}
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
        bold={false}
      />
      <Benchmark
        name={'Montgomery multiplication benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={mont_mul_benchmarks}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Barrett reduction benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={barrett_mul_benchmarks}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Barrett-Domb reduction benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={barrett_domb_mul_benchmarks}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Convert point coordinates to Mont form benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={convert_inputs_into_mont_benchmark}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'BigInts to bytes benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={convert_bigints_to_bytes_benchmark}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'cuzk Serial (Typescript)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={cuzk_typescript_serial}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Point addition algorithm benchmarks'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={add_points_benchmarks}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Transpose (WGSL) - classic algo'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={transpose_wgsl}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'SMVP (WGSL)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={smvp_wgsl}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'SMTVP (WGSL)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={smtvp_wgsl}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Create CSR sparse matrices (precomputation only in TS)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={create_csr_precomputation_benchmark}
        postResult={postResult}
        bold={false}
      />
      <Benchmark
        name={'Create CSR sparse matrices (GPU)'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={create_csr_sparse_matrices_from_points_benchmark}
        postResult={postResult}
        bold={false}
      />
    </div>
  )
};
