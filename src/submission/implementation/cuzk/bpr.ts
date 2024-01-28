import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../../../reference/utils/FieldMath";

export const running_sum_bucket_reduction = (buckets: ExtPointType[]) => {
  const n = buckets.length
  let m = buckets[0]
  let g = m

  //console.log('<rs>')
  //console.log('\tm = buckets[0]; g = m')

  for (let i = 0; i < n - 1; i ++) {
    const idx = n - 1 - i
    //console.log(
      //`\tm = m.add(buckets[${idx}]);` +
      //`g = g.add(m)`
    //)
    const b = buckets[idx]
    m = m.add(b)
    g = g.add(m)
  }
  //console.log('</rs>')

  return g
}

export const parallel_bucket_reduction = (
  buckets: ExtPointType[],
  num_threads = 4,
) => {
  const buckets_per_thread = buckets.length / num_threads
  const bucket_sums: ExtPointType[] = []

  //console.log('<parallel>')
  for (let thread_id = 0; thread_id < num_threads; thread_id ++) {
    //console.log(`\t<thread ${thread_id}>`)

    const idx = thread_id === 0 ? 0 : (num_threads - thread_id) * buckets_per_thread

    let m = buckets[idx]
    let g = m
    //console.log(`\t\tm = buckets[${idx}]; g = m`)

    for (let i = 0; i < buckets_per_thread - 1; i ++) {
      const idx = (num_threads - thread_id) * buckets_per_thread - 1 - i
      //console.log(
        //`\t\tm = m.add(buckets[${idx}]); ` +
        //`g = g.add(m)`
      //)
      const b = buckets[idx]
      m = m.add(b)
      g = g.add(m)
    }

    const s = buckets_per_thread * (num_threads - thread_id - 1)
    if (s > 0) {
      //console.log(`\t\tg.add(m ^ ${s})`)
      g = g.add(m.multiply(BigInt(s)))
    }

    bucket_sums.push(g)
    //console.log('\t</thread>')
  }
  //console.log('</parallel>')
  return bucket_sums
}
