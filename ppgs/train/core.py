import contextlib
import os

import torch
from accelerate import Accelerator
import tqdm

import ppgs

###############################################################################
# Training interface
###############################################################################


def run(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    eval_only=False):
    """Run model training"""
    train(
        dataset=dataset,
        checkpoint_directory=checkpoint_directory,
        output_directory=output_directory,
        log_directory=log_directory,
        eval_only=eval_only)

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
    eval_only=False):
    """Train a model"""

    # Initialize accelerator and get device
    accelerator = Accelerator(
        mixed_precision='fp16',
        even_batches=False,
        # log_with='tensorboard'
    )
    device = accelerator.device

    #################
    # Create models #
    #################

    #TODO config?
    model = ppgs.Model()().to(device)

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
            step,
            epoch
        ) = ppgs.checkpoint.load(
            path[0],
            model,
            optimizer
        )

    else:

        # Train from scratch
        step = 0
        epoch = 0

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(ppgs.RANDOM_SEED)
    train_loader, valid_loader = ppgs.data.loaders(dataset)
    train_batch_sampler = train_loader.batch_sampler
    train_batch_sampler.epoch = epoch
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

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
            accelerator)
        return

    #########
    # Train #
    #########


    # Get total number of steps
    steps = ppgs.NUM_STEPS

    loss_fn = ppgs.train.Loss()

    progress = tqdm.tqdm(
        initial=step,
        total=steps,
        dynamic_ncols=True,
        desc=f'Training {ppgs.CONFIG}')
        
    try:
        model.train()
        while step < steps:

            if not model.training:
                model.train()
                raise ValueError('this should never happen') #TODO remove
            for batch in train_loader:

                # Unpack batch
                # input_ppgs, indices = (item.to(device) for item in batch)
                input_ppgs = batch[0].to(device)
                indices = batch[1].to(device)
                lengths = batch[2].to(device)
                stems = batch[3]

                if frontend is not None:
                    with torch.no_grad():
                        input_ppgs = frontend(input_ppgs).to(torch.float16)

                optimizer.zero_grad()

                # Forward pass
                if ppgs.MODEL == 'convolution':
                    predicted_ppgs = model(input_ppgs)
                else:
                    predicted_ppgs = model(input_ppgs, lengths)

                loss = loss_fn(predicted_ppgs, indices)

                ######################
                # Optimize model #
                ######################

                # Backward pass
                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ppgs.GRAD_INF_CLIP, norm_type='inf')
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ppgs.GRAD_2_CLIP, norm_type=2)
                # for p in model.parameters():
                    # if p.grad is not None and p.grad.norm() >= 1.25:
                    #     print(p.grad.norm(), '2', p.shape, step)#, stems)
                    # if p.grad is not None and p.grad.norm(4) >= 1:
                    #     print(p.grad.norm(4), '4', p.shape, step, stems)
                    # if p.grad is not None and p.grad.abs().max() >= 0.5:
                    #     print(p.grad.abs().max(), 'inf', p.shape, step)#, stems)
                # Update weights

                optimizer.step()

                ###########
                # Logging #
                ###########


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
                            accelerator)

                    ###################
                    # Save checkpoint #
                    ###################

                    if step and step % ppgs.CHECKPOINT_INTERVAL == 0:
                        ppgs.checkpoint.save(
                            model,
                            optimizer,
                            step,
                            epoch,
                            output_directory / f'{step:08d}.pt',
                            accelerator)

                # Update training step count
                if step >= steps:
                    break
                step += 1

                # Update progress bar
                progress.update()

                # torch.cuda.empty_cache()
                # print(torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3))

            # update epoch
            train_batch_sampler.epoch += 1
    except KeyboardInterrupt:
        ppgs.checkpoint.save(
            model,
            optimizer,
            step,
            epoch,
            output_directory / f'{step:08d}.pt',
            accelerator)
    finally:
        # Close progress bar
        progress.close()

    ppgs.checkpoint.save(
        model,
        optimizer,
        step,
        epoch,
        output_directory / f'{step:08d}.pt',
        accelerator)


###############################################################################
# Evaluation
###############################################################################

def evaluate(directory, step, model, frontend, valid_loader, train_loader, accelerator: Accelerator=None):
    """Perform model evaluation"""

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
                input_ppgs, indices, lengths, stems = batch

                if frontend is not None:
                    with torch.no_grad():
                        input_ppgs = frontend(input_ppgs).to(torch.float16)

                # Forward pass
                if ppgs.MODEL == 'convolution':
                    predicted_ppgs = model(input_ppgs)
                else:
                    predicted_ppgs = model(input_ppgs, lengths)

                # Accelerate gather across gpus
                indices = accelerator.pad_across_processes(indices, dim=1, pad_index=-100)
                indices = accelerator.pad_across_processes(indices, dim=0, pad_index=-100)
                indices = accelerator.gather_for_metrics(indices)
                non_pad_batches = torch.argwhere(indices[:, 0] != -100).squeeze(dim=1)
                indices = indices[non_pad_batches]
                predicted_ppgs = accelerator.pad_across_processes(predicted_ppgs, dim=2, pad_index=0)
                predicted_ppgs = accelerator.pad_across_processes(predicted_ppgs, dim=0, pad_index=torch.nan)
                predicted_ppgs = accelerator.gather_for_metrics(predicted_ppgs)
                predicted_ppgs = predicted_ppgs[non_pad_batches]

                # Update metrics
                if accelerator.is_main_process:
                    validation_metrics.update(predicted_ppgs, indices)

                # Finish when we have completed all or enough batches
                if i == ppgs.EVALUATION_BATCHES:
                    break
            for i, batch in enumerate(train_loader):

                # Unpack batch
                input_ppgs, indices, lengths, stems = batch

                if frontend is not None:
                    with torch.no_grad():
                        input_ppgs = frontend(input_ppgs).to(torch.float16)

                # Forward pass
                if ppgs.MODEL == 'convolution':
                    predicted_ppgs = model(input_ppgs)
                else:
                    predicted_ppgs = model(input_ppgs, lengths)

                # Accelerate gather across gpus
                indices = accelerator.pad_across_processes(indices, dim=1, pad_index=-100)
                indices = accelerator.pad_across_processes(indices, dim=0, pad_index=-100)
                indices = accelerator.gather_for_metrics(indices)
                non_pad_batches = torch.argwhere(indices[:, 0] != -100).squeeze(dim=1)
                indices = indices[non_pad_batches]
                predicted_ppgs = accelerator.pad_across_processes(predicted_ppgs, dim=2, pad_index=0)
                predicted_ppgs = accelerator.pad_across_processes(predicted_ppgs, dim=0, pad_index=torch.nan)
                predicted_ppgs = accelerator.gather_for_metrics(predicted_ppgs)
                predicted_ppgs = predicted_ppgs[non_pad_batches]

                # Update metrics
                if accelerator.process_index == 0:
                    training_metrics.update(predicted_ppgs, indices)

                # Finish when we have completed all or enough batches
                if i == ppgs.EVALUATION_BATCHES:
                    break

    # Write to tensorboard
    if accelerator.is_main_process:
        ppgs.write.metrics(directory, step, validation_metrics())
        ppgs.write.metrics(directory, step, training_metrics())

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
