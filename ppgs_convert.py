import coremltools as ct
import torch

filename = input("module file: ")

module = torch.jit.load(filename)

SAMPLE_RATE = 16_000
MAX_DURATION = int(input("Input duration in seconds: "))
HOPSIZE = 160

input_shape = ct.TensorType(
    shape=(
        1, SAMPLE_RATE*MAX_DURATION
    )
)

output_shape = ct.TensorType(
    # shape=(
    #     1, 40, int(SAMPLE_RATE*MAX_DURATION/HOPSIZE)
    # )
)

mlprogram = ct.convert(
    module,
    inputs=[input_shape],
    outputs=[output_shape],
    convert_to='mlprogram',
    debug=True,
)

mlprogram.save(filename.split('.')[0] + '.mlpackage')