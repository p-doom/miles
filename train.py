import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


def train(args):
    configure_logger()
    print("DEBUG: Configured logger.")
    # allocate the GPUs
    pgs = create_placement_groups(args)
    print("DEBUG: Created placement groups.")
    init_tracking(args)
    print("DEBUG: Initialized tracking.")

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    # if args.debug_train_only:
        # args.num_rollout = 1
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    print("DEBUG: Created rollout manager.")

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        print("DEBUG: Offload rollout: loaded weights.")

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()
    print("DEBUG: Updated actor model weights.")

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
        print("DEBUG: Checked weight update equality.")

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            print("DEBUG: Offload rollout: loaded CUDA graph.")
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
        print("DEBUG: Offload rollout: loaded KV cache.")

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))
        print("DEBUG: Performed eval-only rollout.")

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                print("DEBUG: Offloaded critic model.")
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
                    print("DEBUG: Offloaded actor model (after critic-only steps).")
            else:
                actor_model.offload()
                print("DEBUG: Offloaded actor model.")
        else:
            actor_model.clear_memory()
            print("DEBUG: Cleared actor model memory.")
    print("DEBUG: Defined offload_train function.")

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
            print("DEBUG: Onload rollout: loaded weights.")
    print("DEBUG: Defined onload_rollout function.")

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"Starting rollout {rollout_id}.")
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))
            print(f"Performed initial evaluation for rollout {rollout_id}.")

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        print(f"Generated rollout data for rollout {rollout_id}.")

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
            print(f"Offloaded rollout manager for rollout {rollout_id}.")

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            print(f"Started async critic training for rollout {rollout_id}.")
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
                print(f"Started async actor training for rollout {rollout_id} (after critic-only steps).")
            ray.get(critic_train_handle)
            print(f"Finished critic training for rollout {rollout_id}.")
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            print(f"Finished async actor training for rollout {rollout_id}.")

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch):
            print(f"Checking save condition for rollout {rollout_id}.")
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id)
                print(f"Saved actor model for rollout {rollout_id}.")
            if args.use_critic:
                critic_model.save_model(rollout_id)
                print(f"Saved critic model for rollout {rollout_id}.")
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
                print(f"Saved rollout manager data for rollout {rollout_id}.")

        offload_train()
        print(f"Called offload_train for rollout {rollout_id}.")
        onload_rollout()
        print(f"Called onload_rollout for rollout {rollout_id}.")
        actor_model.update_weights()
        print(f"Updated actor model weights for rollout {rollout_id}.")

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
