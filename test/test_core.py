import ppgs


###############################################################################
# Test API
###############################################################################


def test_core(file):
    result = ppgs.from_files_to_files([file])

    # TODO - shape assertion check
    # assert result.shape == ??
    print(result.shape)
