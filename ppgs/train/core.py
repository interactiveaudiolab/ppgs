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
    # no_cache=False,
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
            # no_cache,
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
    # no_cache=False,
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
        lr=ppgs.LEARNING_RATE,
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
    # if no_cache:
    #     raise NotImplementedError()
    # else:
    train_loader, valid_loader = ppgs.data.loaders(dataset, representation=ppgs.REPRESENTATION, reduced_features=True)

    # Prepare FRONTEND
    if ppgs.FRONTEND is not None and callable(ppgs.FRONTEND):
        frontend = ppgs.FRONTEND(device)
    else:
        frontend = None

    if eval_only:
        evaluate(
            log_directory,
            step,
            model,
            frontend,
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

    loss_fn = ppgs.train.Loss()

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
            # input_ppgs, indices = (item.to(device) for item in batch)
            input_ppgs = batch[0].to(device)
            indices = batch[1].to(device)
            lengths = batch[2].to(device)
            stems = batch[3]

            with torch.cuda.amp.autocast():

                if frontend is not None:
                    with torch.no_grad():
                        input_ppgs = frontend(input_ppgs).to(torch.float16)

                # Forward pass
                if ppgs.MODEL == 'convolution':
                    predicted_ppgs = model(input_ppgs)
                else:
                    predicted_ppgs = model(input_ppgs, lengths)

                # Compute loss
                # if step == 35:
                #     import pdb; pdb.set_trace()

                loss = loss_fn(predicted_ppgs, indices)

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            try:
                scaler.scale(loss).backward()
            except:
                import pdb; pdb.set_trace()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ppgs.GRAD_INF_CLIP, norm_type='inf')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ppgs.GRAD_2_CLIP, norm_type=2)
            for p in model.parameters():
                # if p.grad is not None and p.grad.norm() >= 1.25:
                #     print(p.grad.norm(), '2', p.shape, step)#, stems)
                # if p.grad is not None and p.grad.norm(4) >= 1:
                #     print(p.grad.norm(4), '4', p.shape, step, stems)
                if p.grad is not None and p.grad.abs().max() >= 0.5:
                    print(p.grad.abs().max(), 'inf', p.shape, step)#, stems)
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

                    print(torch.cuda.max_memory_allocated(device) / (1024 ** 3), torch.cuda.max_memory_reserved(device) / (1024 ** 3))

                    del loss
                    del predicted_ppgs
                    torch.cuda.empty_cache()

                    evaluate(
                        log_directory,
                        step,
                        model,
                        frontend,
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

            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))

        # update epoch
        train_loader.batch_sampler.epoch += 1

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


def evaluate(directory, step, model, frontend, valid_loader, train_loader, gpu):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Prepare model for evaluation
    model.eval()

    print(f'Evaluating model at step {step}')

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
                    lengths,
                    stems
                ) = (item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)

                if frontend is not None:
                    with torch.no_grad():
                        input_ppgs = frontend(input_ppgs).to(torch.float16)

                # Forward pass
                if ppgs.MODEL == 'convolution':
                    predicted_ppgs = model(input_ppgs)
                else:
                    predicted_ppgs = model(input_ppgs, lengths)

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
                    lengths,
                    stems
                ) = (item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)


                if frontend is not None:
                    with torch.no_grad():
                        input_ppgs = frontend(input_ppgs).to(torch.float16)

                # Forward pass
                if ppgs.MODEL == 'convolution':
                    predicted_ppgs = model(input_ppgs)
                else:
                    predicted_ppgs = model(input_ppgs, lengths)

                # Update metrics
                training_metrics.update(predicted_ppgs, indices)

                # Finish when we have completed all or enough batches
                if i == ppgs.EVALUATION_BATCHES:
                    break

    # print(training_metrics.metrics[0].count, training_metrics.metrics[0].true_positives)
    # return

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
