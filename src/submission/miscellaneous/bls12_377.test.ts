jest.setTimeout(1000000)
import {
    createAffinePoint,
    BASE_FIELD,
} from '../implementation/bls12_377'

describe('BLS12-377', () => {
    const x = BigInt('256948617686061222151099205917657017678095364246733267446961262336245752294716969877488967765059276401456368208320')
    const y = BigInt('227105170010858909588581432763679191613848062092956907254157646977554095359559069986365111542605613311147866154268')
    const z = BigInt('200530079991103180348766713932858124680528544393104247922429137530783589900802654704895259718337700840462356557585')
    const p = createAffinePoint(x, y, z)

    it('createAffinePoint()', () => {
        expect(p.x().toBig().toString()).toEqual(
            '100406495097683584255358201597988016233591504530780704496495127329407856735949558049179479777252726970296785896216'
        )
        expect(p.y().toBig().toString()).toEqual(
            '63807138026163771468611662767681672353158802952448833583661885801557782260041754817382759731402212350042477212809'
        )
        expect(p.z().toBig().toString()).toEqual(
            '1'
        )
    })
    
    it('projective point negation', () => {
        const neg_y = BASE_FIELD - y
        const neg_projective_p = createAffinePoint(x, neg_y, z)

        const neg_p = p.negate()
        expect(neg_p.equals(neg_projective_p)).toBeTruthy()
    })
})
