import { BigIntPoint } from "../../reference/types"
import {
    get_device,
} from '../implementation/cuzk/gpu'

export const print_device_limits = async (
    {}: BigIntPoint[],
    {}: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const device = await get_device()
    console.log(device.limits)
    device.destroy()
    return { x: BigInt(0), y: BigInt(0) }
}
