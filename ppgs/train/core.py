import functools
import shutil

import accelerate
import torch
import tqdm

import ppgs


###############################################################################
# Training
###############################################################################


def run(config, dataset):
    """Train from configuration"""
    # Create output directory
    directory = ppgs.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    train(dataset, directory)


###############################################################################
# Training
###############################################################################


@ppgs.notify.notify_on_finish('training')
def train(dataset, directory=ppgs.RUNS_DIR / ppgs.CONFIG):
    """Train a model"""
    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    # Initialize distributed training
    accelerator = accelerate.Accelerator(
        mixed_precision='fp16',
        even_batches=False)

    #################
    # Create models #
    #################

    if ppgs.FRONTEND is not None and callable(ppgs.FRONTEND):
        frontend = ppgs.FRONTEND(accelerator.device)
    else:
        frontend = None

    model = ppgs.Model()

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters(), lr=ppgs.LEARNING_RATE)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    # Find latest checkpoint
    path = ppgs.checkpoint.latest_path(directory)

    if path is not None:

        # Load model
        model, optimizer, step, epoch = ppgs.checkpoint.load(
            path[0],
            model,
            optimizer)

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
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        valid_loader)

    #########
    # Train #
    #########

    # Get total number of steps
    steps = ppgs.NUM_STEPS

    # Setup progress bar
    progress = tqdm.tqdm(
        initial=step,
        total=steps,
        dynamic_ncols=True,
        desc=f'Training {ppgs.CONFIG}')

    try:
        model.train()
        while step < steps:

            # Update epoch-based random seed
            train_loader.batch_sampler.set_epoch(epoch)

            for batch in train_loader:

                # Unpack batch
                input_representation, indices, lengths = batch

                if frontend is not None:
                    with torch.no_grad():
                        input_representation = frontend(
                            input_representation).to(torch.float16)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                predicted_ppgs = model(input_representation, lengths)

                # Compute loss
                train_loss = loss(predicted_ppgs, indices)

                ##################
                # Optimize model #
                ##################

                # Backward pass
                accelerator.backward(train_loss)

                # Update weights
                optimizer.step()

                ############
                # Evaluate #
                ############

                if step % ppgs.EVALUATION_INTERVAL == 0:

                    # Clear cache to make space for evaluation tensors
                    del train_loss
                    del predicted_ppgs
                    torch.cuda.empty_cache()

                    # Evaluate
                    evaluate_fn = functools.partial(
                        evaluate,
                        directory,
                        step,
                        model,
                        frontend,
                        accelerator=accelerator)
                    with ppgs.inference_context(model):
                        evaluate_fn(train_loader, 'train')
                        evaluate_fn(valid_loader, 'valid')

                ###################
                # Save checkpoint #
                ###################

                if step and step % ppgs.CHECKPOINT_INTERVAL == 0:
                    ppgs.checkpoint.save(
                        model,
                        optimizer,
                        step,
                        epoch,
                        directory / f'{step:08d}.pt',
                        accelerator)

                # Update training step count
                if step >= steps:
                    break
                step += 1

                # Update progress bar
                progress.update()

            # update epoch
            epoch += 1

    except KeyboardInterrupt:

        # Save checkpoint on interrupt
        ppgs.checkpoint.save(
            model,
            optimizer,
            step,
            epoch,
            directory / f'{step:08d}.pt',
            accelerator)

    finally:

        # Close progress bar
        progress.close()

    # Save checkpoint
    ppgs.checkpoint.save(
        model,
        optimizer,
        step,
        epoch,
        directory / f'{step:08d}.pt',
        accelerator)


###############################################################################
# Evaluation
###############################################################################


def evaluate(
    directory,
    step,
    model,
    frontend,
    loader,
    partition,
    accelerator=None):
    """Perform model evaluation"""
    # Setup evaluation metrics
    metrics = ppgs.evaluate.Metrics(partition)

    for i, batch in enumerate(loader):

        # Unpack batch
        input_representation, indices, lengths = batch

        # Maybe encode
        if frontend is not None:
            input_representation = frontend(
                input_representation).to(torch.float16)

        # Forward pass
        predicted_ppgs = model(input_representation, lengths)

        # Gather indices
        indices = accelerator.pad_across_processes(
            indices,
            dim=1,
            pad_index=-100)
        indices = accelerator.pad_across_processes(
            indices,
            dim=0,
            pad_index=-100)
        indices = accelerator.gather_for_metrics(indices)
        non_pad_batches = torch.argwhere(indices[:, 0] != -100).squeeze(dim=1)
        indices = indices[non_pad_batches]

        # Gather PPGs
        predicted_ppgs = accelerator.pad_across_processes(
            predicted_ppgs,
            dim=2,
            pad_index=0)
        predicted_ppgs = accelerator.pad_across_processes(
            predicted_ppgs,
            dim=0,
            pad_index=torch.nan)
        predicted_ppgs = accelerator.gather_for_metrics(predicted_ppgs)
        predicted_ppgs = predicted_ppgs[non_pad_batches]

        # Update metrics
        if accelerator.is_main_process:
            metrics.update(predicted_ppgs, indices)

        # Finish when we have completed all or enough batches
        if i == ppgs.EVALUATION_BATCHES:
            break

    # Write to tensorboard
    if accelerator.is_main_process:
        ppgs.write.metrics(directory, step, metrics())


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
