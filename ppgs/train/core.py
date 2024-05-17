import functools
import math

# import accelerate
import matplotlib
import torch
import torchutil

import ppgs


###############################################################################
# Training
###############################################################################


@torchutil.notify(f'train-{ppgs.CONFIG}')
def train(dataset, directory=ppgs.RUNS_DIR / ppgs.CONFIG, gpu=None):
    """Train a model"""
    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    # Setup accelerator
    # accelerator = accelerate.Accelerator(
    #     mixed_precision='fp16',
    #     even_batches=False)

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #################
    # Create models #
    #################

    if ppgs.FRONTEND is not None and callable(ppgs.FRONTEND):
        # frontend = ppgs.FRONTEND(accelerator.device)
        frontend = ppgs.FRONTEND(device)
    else:
        frontend = None

    model = ppgs.Model().to(device)

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters(), lr=ppgs.LEARNING_RATE)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    # Find latest checkpoint
    path = torchutil.checkpoint.latest_path(directory)

    if path is not None:

        # Load model
        model, optimizer, state = torchutil.checkpoint.load(
            path,
            model,
            optimizer)
        step, epoch = state['step'], state['epoch']

    else:

        # Train from scratch
        step = 0
        epoch = 0

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(ppgs.RANDOM_SEED)
    train_loader = ppgs.data.loader(dataset, 'train')
    valid_loader = ppgs.data.loader(dataset, 'valid')

    ####################
    # Device placement #
    ####################

    # model, optimizer, train_loader, valid_loader = accelerator.prepare(
    #     model,
    #     optimizer,
    #     train_loader,
    #     valid_loader)
    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    #########
    # Train #
    #########

    # Setup progress bar
    progress = torchutil.iterator(
        range(step, ppgs.STEPS),
        f'Training {ppgs.CONFIG}',
        step,
        ppgs.STEPS)

    try:

        while step < ppgs.STEPS:

            # Update epoch-based random seed
            train_loader.batch_sampler.set_epoch(epoch)

            for batch in train_loader:

                # Unpack batch
                input_representation, indices, lengths = (
                    item.to(device) for item in batch)

                if frontend is not None:
                    with torch.no_grad():
                        input_representation = frontend(
                            input_representation
                        ).to(torch.float16)

                with torch.autocast(device.type):

                    # Forward pass
                    predicted_ppgs = model(input_representation, lengths)

                    # Compute loss
                    train_loss = loss(predicted_ppgs, indices)

                ##################
                # Optimize model #
                ##################

                # Zero gradients
                optimizer.zero_grad()

                # Backward pass
                # accelerator.backward(train_loss)
                scaler.scale(train_loss).backward()

                # Monitor gradient statistics
                gradient_statistics = torchutil.gradients.stats(model)
                torchutil.tensorboard.update(
                    directory,
                    step,
                    scalars=gradient_statistics)

                # Maybe perform gradient clipping
                if (
                    ppgs.GRADIENT_CLIP_THRESHOLD_INF is not None or
                    ppgs.GRADIENT_CLIP_THRESHOLD_L2 is not None
                ):

                    # Unscale gradients
                    scaler.unscale_(optimizer)

                    if ppgs.GRADIENT_CLIP_THRESHOLD_L2 is not None:

                        # Compare gradient norm to threshold
                        grad_norm = gradient_statistics['gradients/norm']
                        if grad_norm > ppgs.GRADIENT_CLIP_THRESHOLD_L2:

                            # Clip
                            # accelerator.clip_grad_norm_(
                            #     model.parameters(),
                            #     ppgs.GRADIENT_CLIPPING_THRESHOLD,
                            #     norm_type=2.0)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                ppgs.GRADIENT_CLIP_THRESHOLD_L2,
                                norm_type=2.0)

                    if ppgs.GRADIENT_CLIP_THRESHOLD_INF is not None:

                        # Compare maximum gradient to threshold
                        max_grad = max(
                            gradient_statistics['gradients/max'],
                            math.abs(gradient_statistics['gradients/min']))
                        if max_grad > ppgs.GRADIENT_CLIP_THRESHOLD_INF:

                            # Clip
                            # accelerator.clip_grad_norm_(
                            #     model.parameters(),
                            #     ppgs.GRADIENT_CLIPPING_THRESHOLD,
                            #     norm_type='inf')
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                ppgs.GRADIENT_CLIP_THRESHOLD_INF,
                                norm_type='inf')

                # Update weights
                # optimizer.step()
                scaler.step(optimizer)

                # Update gradient scaler
                scaler.update()

                ############
                # Evaluate #
                ############

                if step % ppgs.EVALUATION_INTERVAL == 0:

                    # Log VRAM utilization
                    # index = accelerator.device.index
                    # print(torch.cuda.memory_summary(device.index))
                    torchutil.tensorboard.update(
                        directory,
                        step,
                        scalars=torchutil.cuda.utilization(device, 'MB'))

                    # Clear cache to make space for evaluation tensors
                    del train_loss
                    del predicted_ppgs
                    torch.cuda.empty_cache()

                    # Evaluate
                    evaluation_steps = (
                        None if step == ppgs.STEPS
                        else ppgs.DEFAULT_EVALUATION_STEPS)
                    evaluate_fn = functools.partial(
                        evaluate,
                        directory,
                        step,
                        model,
                        gpu,
                        frontend,
                        # accelerator=accelerator,
                        evaluation_steps=evaluation_steps)
                    with torchutil.inference.context(model):
                        # evaluate_fn(train_loader, 'train')
                        evaluate_fn(valid_loader, 'valid')

                ###################
                # Save checkpoint #
                ###################

                if step and step % ppgs.CHECKPOINT_INTERVAL == 0:
                    torchutil.checkpoint.save(
                        directory / f'{step:08d}.pt',
                        model,
                        optimizer,
                        # accelerator=accelerator,
                        step=step,
                        epoch=epoch)

                # Update training step count
                if step >= ppgs.STEPS:
                    break
                step += 1

                # Update progress bar
                progress.update()

            # update epoch
            epoch += 1

    except KeyboardInterrupt:

        # Save checkpoint on interrupt
        torchutil.checkpoint.save(
            directory / f'{step:08d}.pt',
            model,
            optimizer,
            # accelerator=accelerator,
            step=step,
            epoch=epoch)

    finally:

        # Close progress bar
        progress.close()

    # Save checkpoint
    torchutil.checkpoint.save(
        directory / f'{step:08d}.pt',
        model,
        optimizer,
        # accelerator=accelerator,
        step=step,
        epoch=epoch)


###############################################################################
# Evaluation
###############################################################################


def evaluate(
    directory,
    step,
    model,
    gpu,
    frontend,
    # accelerator=None,
    loader,
    partition,
    evaluation_steps=None
):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Setup evaluation metrics
    metrics = ppgs.evaluate.Metrics(partition)

    for i, batch in enumerate(loader):

        # Unpack batch
        input_representation, indices, lengths = (
            item.to(device) for item in batch
        )

        # Maybe encode
        if frontend is not None:
            input_representation = frontend(
                input_representation
            ).to(torch.float16)

        # Forward pass
        predicted_ppgs = model(input_representation, lengths)

        # Gather indices
        # indices = accelerator.pad_across_processes(
        #     indices,
        #     dim=1,
        #     pad_index=-100)
        # indices = accelerator.pad_across_processes(
        #     indices,
        #     dim=0,
        #     pad_index=-100)
        # indices = accelerator.gather_for_metrics(indices)
        # non_pad_batches = torch.argwhere(indices[:, 0] != -100).squeeze(dim=1)
        # indices = indices[non_pad_batches]

        # Gather PPGs
        # predicted_ppgs = accelerator.pad_across_processes(
        #     predicted_ppgs,
        #     dim=2,
        #     pad_index=0)
        # predicted_ppgs = accelerator.pad_across_processes(
        #     predicted_ppgs,
        #     dim=0,
        #     pad_index=torch.nan)
        # predicted_ppgs = accelerator.gather_for_metrics(predicted_ppgs)
        # predicted_ppgs = predicted_ppgs[non_pad_batches]

        # Update metrics
        # if accelerator.is_main_process:
        metrics.update(predicted_ppgs, indices)

        # Stop when we exceed some number of batches
        if evaluation_steps is not None and i + 1 == evaluation_steps:
            break

    # Write to tensorboard
    scalars, figures = {}, {}
    for key, val in metrics().items():
        if isinstance(val, matplotlib.figure.Figure):
            figures[f'{partition}/{key}'] = val
        else:
            scalars[f'{partition}/{key}'] = val
    torchutil.tensorboard.update(
        directory,
        step,
        figures=figures,
        scalars=scalars)


###############################################################################
# Loss
###############################################################################


def loss(input, target, reduction='mean'):
    """Loss function"""
    if ppgs.CLASS_BALANCED:
        if not hasattr(loss, 'weights'):
            loss.weights = ppgs.load.phoneme_weights(input.device)
        return torch.nn.functional.cross_entropy(
            input,
            target,
            loss.weights,
            reduction=reduction)
    return torch.nn.functional.cross_entropy(
        input,
        target,
        reduction=reduction)
