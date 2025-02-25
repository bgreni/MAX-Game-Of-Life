import compiler
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList

alias ON = 255
alias OFF = 0

@compiler.register('conway', num_dps_outputs=1)
struct Conway[wrap: Bool]:
    @staticmethod
    fn execute[target: StringLiteral](
        out: ManagedTensorSlice,
        x: ManagedTensorSlice[type=out.type, rank=out.rank],
        ctx: DeviceContextPtr
    ):
        var shape = x.shape()

        # shape gets corrupted for large tensors if I don't copy it in here?
        @__copy_capture(shape)
        @parameter
        @always_inline
        fn conway_elementwise[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:

            @always_inline
            @parameter
            fn check_pos(owned xl: Int, owned yl: Int) -> Scalar[x.type]:
                
                @parameter
                if not wrap:
                    if xl < 0 or yl < 0 or xl >= shape[0] or yl >= shape[1]:
                        return OFF

                var x_ind: Int
                var y_ind: Int

                @parameter
                if wrap:
                    xl %= shape[0]
                    yl %= shape[1]
                    
                return x[IndexList[x.rank](xl, yl)] & 1

            var row = idx[0]
            var col = idx[1]

            var total: Scalar[x.type] = 0

            total += check_pos(row - 1, col - 1)
            total += check_pos(row - 1, col)
            total += check_pos(row - 1, col + 1)
            total += check_pos(row, col - 1)
            total += check_pos(row, col + 1)
            total += check_pos(row + 1, col - 1)
            total += check_pos(row + 1, col)
            total += check_pos(row + 1, col + 1)

            var curr = x[idx] & 1
    
            return ((ON * Scalar[x.type](total == 2 or total == 3)) * curr) 
                | ((ON * Scalar[x.type](total == 3)) * (~curr) & 1)

        foreach[conway_elementwise, target=target, simd_width=1](out, ctx)

    @staticmethod
    fn shape(
        x: ManagedTensorSlice,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"