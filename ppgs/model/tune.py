import torch

import ppgs


def get_max_frames(model: torch.nn.Module, loss_fn, optimizer: torch.optim, scaler, device, overhead: int = 2e8):
    """Test the max number of frames the current model can handle, with given overhead"""

    overhead_tensor = torch.rand((int(overhead),), device=device)
    overhead_tensor.max()
    # print("overhead + model allocated:", torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    # print("overhead + model reserved:", torch.cuda.max_memory_reserved(device) / (1024 ** 3))
    model.to(device)

    magnitudes = range(2, 20)
    done=False
    with torch.set_grad_enabled(True), torch.cuda.amp.autocast():
        for magnitude in magnitudes:
            num_frames = 10 ** magnitude
            for i in range(0, magnitude + 1):
                print(torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                length = 10 ** i
                batch_size = num_frames // length
                # print(num_frames, length, batch_size)
                try:
                    fake_input = torch.zeros((batch_size, ppgs.INPUT_CHANNELS, length),device=device)
                    # print("input size in memory:", fake_input.element_size()*fake_input.nelement() / (1024 ** 3))
                    fake_lengths = torch.full((batch_size,), length, device=device, dtype=torch.long)
                    output = model(fake_input, fake_lengths)
                    loss = loss_fn(output, torch.zeros((batch_size, length), dtype=torch.long, device=device))
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                except torch.cuda.OutOfMemoryError:
                    print(f'OOM with num frames {num_frames}, batch size {batch_size}, length {length}')
                    done=True
                    break
                except ValueError:
                    print(f'model cannot run with batch size {batch_size} and length {length}')
                    break
                finally:
                    try:
                        del fake_input
                        del fake_lengths
                        del output
                        del loss
                        torch.cuda.empty_cache()
                    except UnboundLocalError:
                        pass
                    optimizer.zero_grad()
            if done:
                break
    magnitude = magnitude - 1
    print("magnitude:", magnitude)

    multiples = range(1, 10)
    done = False
    with torch.set_grad_enabled(True), torch.cuda.amp.autocast():
        for multiple in multiples:
            num_frames = (10 ** magnitude) * multiple
            for i in range(0, magnitude + 1):
                # print("start:", torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                length = 10 ** i
                batch_size = num_frames // length
                # print(num_frames, length, batch_size)
                try:
                    fake_input = torch.zeros((batch_size, ppgs.INPUT_CHANNELS, length),device=device)
                    fake_lengths = torch.full((batch_size,), length, device=device, dtype=torch.long)
                    # print("input size in memory:", fake_input.element_size()*fake_input.nelement() / (1024 ** 3))
                    # # print("after inputs:", torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                    output = model(fake_input, fake_lengths)
                    total_bytes = 0
                    for parameter in model.parameters():
                        total_bytes += parameter.grad.element_size()*parameter.grad.nelement()
                    # print("gradients size in memory:", total_bytes / (1024 ** 3))
                    # print("output size in memory:", output.element_size()*output.nelement() / (1024 ** 3))
                    # print("after output:", torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                    loss = loss_fn(output, torch.zeros((batch_size, length), dtype=torch.long, device=device))
                    # print("after loss:", torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    # print("after backward:", torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                    scaler.step(optimizer)
                    scaler.update()
                    # print("end:", torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))
                except torch.cuda.OutOfMemoryError:
                    print(f'OOM with num frames {num_frames}, batch size {batch_size}, length {length}')
                    done=True
                    break
                except ValueError:
                    print(f'model cannot run with batch size {batch_size} and length {length}')
                    break
                finally:
                    try:
                        del fake_input
                        del fake_lengths
                        del output
                        del loss
                    except UnboundLocalError:
                        pass
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            if done:
                break
    multiple=multiple - 1
    print("multiple:", multiple)
    print("Approximate max number of frames:", (10 ** magnitude)*multiple)
    print("max allocated:", torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    print("max reserved:", torch.cuda.max_memory_reserved(device) / (1024 ** 3))


if __name__ == '__main__':
    loss_fn = torch.nn.functional.cross_entropy
    scaler = torch.cuda.amp.GradScaler()
    device = 'cuda:1'
    model = ppgs.Model()().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=[.80, .99],
        eps=1e-9)
    max_frames = get_max_frames(
        model,
        loss_fn,
        optimizer,
        scaler,
        device,
    )
