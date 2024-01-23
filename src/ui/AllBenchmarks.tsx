import React, { useEffect, useState } from 'react';
import { Benchmark } from './Benchmark';
import { bigIntToU32Array, generateRandomFields } from '../reference/webgpu/utils';
import { BigIntPoint, U32ArrayPoint } from '../reference/types';
import { wasm_compute_bls12_377_msm } from '../reference/reference';
import { compute_msm } from '../submission/submission';
import CSVExportButton from './CSVExportButton';
import { TestCaseDropDown } from './TestCaseDropDown';
import { PowersTestCase, TestCase, loadTestCase } from '../test-data/testCases';
//import { full_benchmarks } from '../submission/miscellaneous/full_benchmarks'

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
      const x = BigInt('111871295567327857271108656266735188604298176728428155068227918632083036401841336689521497731900230387779623820740');
      const y = BigInt('76860045326390600098227152997486448974650822224305058012700629806287380625419427989664237630603922765089083164740');
      const z = BigInt('1');
      const point: BigIntPoint = {x, y, z};
      const newPoints = Array(inputSize).fill(point);
      setBaseAffineBigIntPoints(newPoints);
      const newU32Points = newPoints.map((point) => {
        return {
          x: bigIntToU32Array(point.x),
          y: bigIntToU32Array(point.y),
          z: bigIntToU32Array(point.z),
        }});
      setU32Points(newU32Points);

      const newScalars = generateRandomFields(inputSize);
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
        name={'Aleo Wasm: Single Thread'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={wasm_compute_bls12_377_msm}
        postResult={postResult}
      />
      <Benchmark
        name={'Your MSM'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={compute_msm}
        postResult={postResult}
        bold={true}
      />

      {/*<div className="flex items-left">
        <div className={`text-gray-800 w-40 px-2 font-bold'`}>{name}</div> 
          <h1>Miscellaneous benchmarks and tests</h1>
          <br />
          <br />
      </div>

      <Benchmark
        name={'Print device limits'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={print_device_limits}
        postResult={postResult}
        bold={false}
      /> */}
      {/* <Benchmark
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
      /> */}
    </div>
  )
};
