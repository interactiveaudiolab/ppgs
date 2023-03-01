import contextlib
import functools
import os

import torch
import tqdm

import ppgs
from ppgs.evaluate import visualize

###############################################################################
# Training interface
###############################################################################


def run(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus=None,
    eval_only=False):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus)
        torch.multiprocessing.spawn(
            train_ddp,
            args=args,
            nprocs=len(gpus),
            join=True)

    else:
        # Single GPU or CPU training
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            None if gpus is None else gpus[0],
            eval_only)

    # Return path to model checkpoint
    return ppgs.checkpoint.latest_path(output_directory)


###############################################################################
# Training
###############################################################################


def train(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpu=None,
    eval_only=False):
    """Train a model"""
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')


    #################
    # Create models #
    #################

    #TODO config?
    model = ppgs.Model()().to(device)

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=[.80, .99],
        eps=1e-9)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = ppgs.checkpoint.latest_path(
        checkpoint_directory,
        '*.pt'),

    # For some reason, returning None from latest_path returns (None,)
    path = None if path == (None,) else path

    if path is not None:

        # Load model
        (
            model,
            optimizer,
            step
        ) = ppgs.checkpoint.load(
            path[0],
            model,
            optimizer
        )

    else:

        # Train from scratch
        step = 0

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(ppgs.RANDOM_SEED)
    train_loader, valid_loader = ppgs.data.loaders(dataset, representation=ppgs.REPRESENTATION)


    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=ppgs.LEARNING_RATE_DECAY,
        last_epoch=step // len(train_loader.dataset) if step else -1)
    scheduler = scheduler_fn(optimizer)


    if eval_only:
        evaluate(
            log_directory,
            step,
            model,
            valid_loader,
            train_loader,
            gpu)
        return

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Get total number of steps
    steps = ppgs.NUM_STEPS

    # Setup progress bar
    if not rank:
        progress = tqdm.tqdm(
            initial=step,
            total=steps,
            dynamic_ncols=True,
            desc=f'Training {ppgs.CONFIG}')
    while step < steps:

        model.train()
        for batch in train_loader:

            # Unpack batch
            input_ppgs, indices = (item.to(device) for item in batch[:2])

            with torch.cuda.amp.autocast():

                # Forward pass
                predicted_ppgs = model(input_ppgs)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    predicted_ppgs,
                    indices)

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(loss).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                if step % ppgs.LOG_INTERVAL == 0:

                    # Log loss
                    scalars = {
                        'train/loss': loss,
                        'learning_rate': optimizer.param_groups[0]['lr']}
                    ppgs.write.scalars(log_directory, step, scalars)

                    # Log visualizations
                    # visualization_batch = batch[:ppgs.VISUALIZATION_SAMPLES]
                    # audio_filenames = [f + '.wav' for f in visualization_batch]

                ############
                # Evaluate #
                ############

                if step % ppgs.EVALUATION_INTERVAL == 0:

                    evaluate(
                        log_directory,
                        step,
                        model,
                        valid_loader,
                        train_loader,
                        gpu)

                ###################
                # Save checkpoint #
                ###################

                if step and step % ppgs.CHECKPOINT_INTERVAL == 0:
                    ppgs.checkpoint.save(
                        model,
                        optimizer,
                        step,
                        output_directory / f'{step:08d}.pt')

            # Update training step count
            if step >= steps:
                break
            step += 1

            # Update progress bar
            if not rank:
                progress.update()

        # Update learning rate every epoch
        scheduler.step()

    # Close progress bar
    if not rank:
        progress.close()

    # Save final model
    ppgs.checkpoint.save(
        model,
        optimizer,
        step,
        output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, valid_loader, train_loader, gpu):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision
        with torch.cuda.amp.autocast():

            # Setup evaluation metrics
            training_metrics = ppgs.evaluate.Metrics('training')
            validation_metrics = ppgs.evaluate.Metrics('validation')

            for i, batch in enumerate(valid_loader):

                # Unpack batch
                (
                    input_ppgs,
                    indices,
                    alignments,
                    word_breaks,
                    waveforms,
                    stems
                ) = (item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)

                # Forward pass
                predicted_ppgs = model(input_ppgs)

                # Update metrics
                validation_metrics.update(predicted_ppgs, indices)

                # Finish when we have completed all or enough batches
                if i == ppgs.EVALUATION_BATCHES:
                    break

            for i, batch in enumerate(train_loader):

                # Unpack batch
                (
                    input_ppgs,
                    indices,
                    alignments,
                    word_breaks,
                    waveforms,
                    stems
                ) = (item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)

                # Forward pass
                predicted_ppgs = model(input_ppgs)

                # Update metrics
                training_metrics.update(predicted_ppgs, indices)

                # Finish when we have completed all or enough batches
                if i == ppgs.EVALUATION_BATCHES:
                    break

    print(training_metrics.metrics[0].count, training_metrics.metrics[0].true_positives)
    return

    # Write to tensorboard
    ppgs.write.scalars(directory, step, validation_metrics())
    ppgs.write.scalars(directory, step, training_metrics())

    # Prepare model for training
    model.train()


###############################################################################
# Distributed data parallelism
###############################################################################

#TODO look in updated template
def train_ddp(rank, dataset, checkpoint_directory, output_directory, log_directory, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, checkpoint_directory, output_directory, log_directory, gpus[rank])


@contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()
