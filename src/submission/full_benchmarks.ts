import { BigIntPoint } from "../reference/types"
import { PowersTestCase, loadTestCase } from '../test-data/testCases';
import { cuzk_gpu } from './cuzk/cuzk_gpu'

export const full_benchmarks = async (
    {}: BigIntPoint[], // Use {} to prevent Typescript warnings about unused parameters
    {}: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const DELAY = 100
    const NUM_RUNS = 4

    const START_POWER: PowersTestCase = 16
    const END_POWER: PowersTestCase = 20

    const all_results: any = {}

    for (let power = START_POWER; power <= END_POWER; power ++) {
        const testcase = await loadTestCase(power)

        /*
          Iterate test cases of powers 16 to 20 inclusive
          For each test case:
            - inject some random variables into each shader to force recompilation
            - measure the first run
            - measure NUM_RUNS more runs
            - report the time taken for all runs, the average, and the average
              excluding the first (compile) run in JSON and Markdown
        */

        // TODO: inject random variables in to the shader

        // Measure the first run
        const first_run_start = Date.now()
        const msm = await cuzk_gpu(testcase.baseAffinePoints, testcase.scalars, false)
        const first_run_elapsed = Date.now() - first_run_start

        const expected = testcase.expectedResult
        if (msm.x !== expected.x || msm.y !== expected.y) {
            console.error(`WARNING: the result of cuzk_gpu is incorrect for 2^${power}`)
        }

        await delay(DELAY)

        const results: {
            first_run_elapsed: number,
            subsequent_runs: number[],
            full_average: number,
            subsequent_average: number,
        } = {
            first_run_elapsed,
            subsequent_runs: [],
            full_average: 0,
            subsequent_average: 0,
        }

        for (let i = 0; i < NUM_RUNS; i ++) {
            const start = Date.now()
            await cuzk_gpu(testcase.baseAffinePoints, testcase.scalars, false)
            const elapsed = Date.now() - start

            results.subsequent_runs.push(elapsed)
            await delay(DELAY)
        }

        let subsequent_total = 0
        for (let i = 0; i < results.subsequent_runs.length; i ++) {
            subsequent_total += results.subsequent_runs[i]
        }

        results['full_average'] = Math.round((results.first_run_elapsed + subsequent_total) /
            (1 + results.subsequent_runs.length))

        results['subsequent_average'] = Math.round(subsequent_total /
            results.subsequent_runs.length)

        all_results[power] = results
    }

    let header = `| MSM size | 1st run |`
    for (let i = 0; i < NUM_RUNS; i ++) {
        header += ` #${i + 1} |`
    }

    header += ` Average (incl 1st) | Average (excl 1st) |`
    header += `\n|-|-|-|-|\n`
    let body = ``

    for (let power = START_POWER; power <= END_POWER; power ++) {
        const result = all_results[power]
        let md = `| 2^${power} | ${result.first_run_elapsed } |`
        for (let i = 0; i < result.subsequent_runs.length; i ++) {
            const r = result.subsequent_runs[i]
            md += ` ${r} |`
        }

        body += md + ` **${result.full_average}** | **${result.subsequent_average}** |\n`
    }

    console.log(header + body.trim())
    return { x: BigInt(0), y: BigInt(1) }
}

export const delay = (duration: number) => {
    return new Promise( resolve => setTimeout(resolve, duration))
}
