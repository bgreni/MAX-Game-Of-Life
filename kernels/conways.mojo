import compiler
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList

alias ON = 255
alias OFF = 0

@compiler.register('conway', num_dps_outputs=1)
struct Conway:
    @staticmethod
    fn execute[target: StringLiteral](
        out: ManagedTensorSlice,
        x: ManagedTensorSlice[type=out.type, rank=out.rank],
        ctx: DeviceContextPtr
    ):
        var shape = x.shape()
        @parameter
        @always_inline
        fn conway_elementwise[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            @always_inline
            @parameter
            fn check_pos(xl: Int, yl: Int) -> Int:
                if xl < 0 or yl < 0 or xl >= shape[0] or yl >= shape[1]:
                    return 0

                return 1 if x[IndexList[x.rank](xl, yl)] == 255 else 0

            var row = idx[0]
            var col = idx[1]

            var total = 0

            total += check_pos(row - 1, col - 1)
            total += check_pos(row - 1, col)
            total += check_pos(row - 1, col + 1)
            total += check_pos(row, col - 1)
            total += check_pos(row, col + 1)
            total += check_pos(row + 1, col - 1)
            total += check_pos(row + 1, col)
            total += check_pos(row + 1, col + 1)


            var curr = x[idx]

            if curr:
                if total == 2 or total == 3:
                    return ON
            else:
                if total == 3:
                    return ON
            return OFF


        foreach[conway_elementwise, target=target, simd_width=1](out, ctx)

    @staticmethod
    fn shape(
        x: ManagedTensorSlice,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"