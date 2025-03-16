import compiler
from max.tensor import OutputTensor, InputTensor, foreach, ManagedTensorSlice
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList

alias ON = 255
alias OFF = 0

@compiler.register('conway')
struct Conway[wrap: Bool]:
    @staticmethod
    fn execute[target: StringLiteral](
        out: OutputTensor,
        x: InputTensor[type=out.type, rank=out.rank],
        ctx: DeviceContextPtr
    ) raises:
        var shape = x.shape()

        alias CellType = Scalar[x.type]

        @__copy_capture(shape)
        @parameter
        @always_inline
        fn conway_elementwise[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:

            @always_inline
            @parameter
            fn check_pos(owned yl: Int, owned xl: Int) -> CellType:
                
                @parameter
                if not wrap:
                    if xl < 0 or yl < 0 or yl >= shape[0] or xl >= shape[1]:
                        return OFF

                @parameter
                if wrap:
                    yl %= shape[0]
                    xl %= shape[1]
                    
                return x[IndexList[x.rank](yl, xl)] & 1

            var row = idx[0]
            var col = idx[1]

            var total: CellType = 0

            total += check_pos(row - 1, col - 1)
            total += check_pos(row - 1, col)
            total += check_pos(row - 1, col + 1)
            total += check_pos(row, col - 1)
            total += check_pos(row, col + 1)
            total += check_pos(row + 1, col - 1)
            total += check_pos(row + 1, col)
            total += check_pos(row + 1, col + 1)

            var curr_is_on = x[idx] & 1
    
            return ((ON * CellType(
                (CellType(total == 2) * curr_is_on) | CellType(total == 3))
            ))

        foreach[conway_elementwise, target=target, simd_width=1](out, ctx)

    @staticmethod
    fn shape(
        x: InputTensor
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"