import argparse
import os
import sys
import argparse
import torch
import torchvision
import models
import utils
import dataset

from supervision import *
from exporters import *
from utils import *

def parse_arguments(args):
    usage_text = (
        "Deep Depth Denoising Training."
        "Usage:  python train.py [options],"
        "   with [options]: (as described below)"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # durations
    parser.add_argument('-e',"--epochs", type = int, help = "Train for a total number of <epochs> epochs.")
    parser.add_argument('-b',"--batch_size", type = int, help = "Train with a <batch_size> number of samples each train iteration.")
    parser.add_argument("--test_batch_size", default=1, type = int, help = "Test with a <batch_size> number of samples each test iteration.")    
    parser.add_argument('-d','--disp_iters', type=int, default=50, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('-c','--checkpoint_iters', type=int, default=1000, help='Save checkpoint (i.e. weights & optimizer) every <checkpoint_iters> iterations.')
    parser.add_argument('-t','--test_iters', type=int, default=1000, help='Test model every <test_iters> iterations.')
    parser.add_argument('--max_test_iters', type=int, default=100, help='Maximum test iterations to perform each test run.')
    # paths
    parser.add_argument("--train_path", type = str, help = "Path to the training folder containing the files")
    parser.add_argument("--test_path", type = str, help = "Path to the testing folder containing the files")    
    # model
    parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--dilation', type=int, default=1, help='Dilation value for bottleneck convolutions')
    parser.add_argument('--normalization', type=str, default="elu", help='Choose in-model activation normalization. Supported types elu or batch_norm')
    parser.add_argument('--ndf', type=int, default=8, help='Constant values used to define input and output channels at nn layers')
    parser.add_argument('--upsample_type', default="nearest", type=str, help='Model selection argument.')
    # optimization
    parser.add_argument('-o','--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
    parser.add_argument("--opt_state", type = str, help = "Path to stored optimizer state file (for continuing training)")
    parser.add_argument('-l','--lr', type=float, default=0.0002, help='Optimization Learning Rate.')
    parser.add_argument('-m','--momentum', type=float, default=0.9, help='Optimization Momentum.')
    parser.add_argument('--momentum2', type=float, default=0.999, help='Optimization Second Momentum (optional, only used by some optimizers).')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization Weight Decay.')    
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    # other
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument("--device_list",nargs="*", type=str, default = ["M72e","M72h","M72i","M72j"], help = "List of device names to be loaded")
    parser.add_argument("--super_list",nargs="*", type=str, default = ["M72e","M72h","M72i","M72j"], help = "List of device names to be used as supervision")
    parser.add_argument("--visdom", type=str, nargs='?', default=None, const="127.0.0.1", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument("--visdom_iters", type=int, default=400, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument("--seed", type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
    # network specific params
    parser.add_argument("--photo_w", type=float, default=0.85, help = "Photometric loss weight.")
    parser.add_argument("--depth_reg_w", type=float, default=0.1, help = "Depth regularization weight.")
    parser.add_argument("--normal_reg_w", type=float, default=0.05, help = "Surface smoothness weight.")
    # data handlers
    parser.add_argument("--depth_thres", type=float, default=3.0, help = "Depth threshold - depth clipping.")
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device, visualizer, model_params = utils.initialize(args)
    # set model type and init weights
    model = models.get_model(model_params)
    utils.init.initialize_weights(model, args.weight_init)
    if (len(gpus) > 1):        
        model = torch.nn.parallel.DataParallel(model, gpus)
    model = model.to(device)
    # init optimizer
    optimizer = utils.init_optimizer(model, args)
    # training data loader
    train_data_params = dataset.dataloader.DataLoaderParams(\
        root_path=args.train_path, device_list=args.device_list,\
        device_repository_path=args.train_path, depth_threshold=args.depth_thres) 
    train_data_iterator = dataset.dataloader.DataLoad(train_data_params)
    train_set = torch.utils.data.DataLoader(train_data_iterator,\
        batch_size = args.batch_size, shuffle=True,\
        num_workers = args.batch_size // len(gpus), pin_memory=False)
    # validation data loader
    test_data_params = dataset.dataloader.DataLoaderParams(root_path=args.test_path,\
        device_list=["M72j"], device_repository_path=args.train_path,\
        depth_threshold=args.depth_thres) 
    test_data_iterator = dataset.dataloader.DataLoad(test_data_params)
    test_set = torch.utils.data.DataLoader(test_data_iterator,\
        batch_size = args.test_batch_size, shuffle=True,\
        num_workers = args.test_batch_size // len(gpus), pin_memory=False)
    print("Data size : {0} | Test size : {1}".format(train_data_iterator.__len__(),test_data_iterator.__len__()))
    # loss definition
    total_loss = AverageMeter()
    running_photo_loss = AverageMeter()
    running_depth_reg_loss = AverageMeter()
    running_surface_smooth_loss = AverageMeter()

    uv_grid = create_image_domain_grid(640, 360).to(device).expand(args.batch_size, -1, -1, -1)
    fov_w = fov_weights(uv_grid).to(device).expand(args.batch_size, -1, -1, -1)
    # training
    model.train()
    iteration_counter = 0
    for epoch in range(args.epochs):
        print("Training | Epoch: {}".format(epoch))
        utils.opt.adjust_learning_rate(optimizer, epoch)
        for batch_id, batch in enumerate(train_set):            
            b, c, h, w = next(iter(batch.items()))[1]["depth"].size()
            if b < args.batch_size:
                continue
            # loss init
            active_loss = torch.tensor(0.0).to(device)
            photo_loss = 0.0
            depth_reg_loss = 0.0
            normal_loss = 0.0

            optimizer.zero_grad()
            processed = get_processed_info(batch, model, device,\
                args.super_list, threshold=args.depth_thres)
            add_3d_info(processed, batch, uv_grid)
            add_forward_rendering_info(processed, uv_grid,\
                depth_threshold=args.depth_thres, fov_w=fov_w)           

            if args.photo_w > 0.0:
                photo_l, photo_l_map = robust_photometric_supervision_splat(processed)                
                active_loss += args.photo_w * photo_l
                photo_loss += args.photo_w * photo_l
            if args.depth_reg_w > 0.0:
                d_reg_l, d_reg_l_map = depth_regularisation(processed)
                active_loss += args.depth_reg_w * d_reg_l
                depth_reg_loss += args.depth_reg_w * d_reg_l
            if args.normal_reg_w > 0.0:
                add_normal_info(processed)
                normal_l, normal_l_map = surface_smoothness_prior(processed)
                active_loss += args.normal_reg_w * normal_l
                normal_loss += args.normal_reg_w * normal_l
            active_loss.backward()
            optimizer.step()

            # loss update
            total_loss.update(active_loss)
            running_depth_reg_loss.update(depth_reg_loss)
            running_photo_loss.update(photo_loss)
            running_surface_smooth_loss.update(normal_loss)
            iteration_counter += b

            # validation
            if (iteration_counter + 1) % args.test_iters <= args.batch_size:
                model.eval()
                test_loss = 0
                counter = 0
                with torch.no_grad():
                    for test_batch_id , test_batch in enumerate(test_set):
                        datum = next(iter(test_batch.values()))
                        b, c, h, w = datum["depth"].shape
                        counter += b
                        if counter > args.max_test_iters:
                            break
                        uv_grid_t = create_image_domain_grid(w, h).to(device)                        
                        for attribute in datum:
                            datum[attribute] = datum[attribute].to(device)
                        
                        original_depth = datum["depth"]
                        original_mask, __count = get_mask(original_depth, max_threshold=args.depth_thres)
                        # predict and mask depth
                        predicted_depth, _ = model(original_depth, original_mask)
                        masked_predicted_depth = predicted_depth * original_mask
                        # save point clouds
                        intrinsics_inv = datum["intrinsics_inv"]                        
                        source_points3d = deproject_depth_to_points(original_depth, uv_grid_t, intrinsics_inv)
                        predicted_points3d = deproject_depth_to_points(predicted_depth, uv_grid_t, intrinsics_inv)
                        masked_predicted_points3d = deproject_depth_to_points(masked_predicted_depth, uv_grid_t, intrinsics_inv)

                        save_ply(os.path.join(args.test_path, str(iteration_counter) + "_original_#.ply"), source_points3d, 1000.0, color='red')
                        save_ply(os.path.join(args.test_path, str(iteration_counter) + "_denoised_#.ply"), predicted_points3d, 1000.0, color='green')
                        save_ply(os.path.join(args.test_path, str(iteration_counter) + "_masked_denoised_#.ply"), masked_predicted_points3d, 1000.0, color='blue')

                        source_n3d = calculate_normals(source_points3d)
                        masked_predicted_n3d = calculate_normals(masked_predicted_points3d)

                        save_depth_from_3d(os.path.join(args.test_path, str(iteration_counter) + "_depth_#.png"), source_points3d, 1000.0)
                        save_normals(os.path.join(args.test_path, str(iteration_counter) + "_normals_#.png"), source_n3d)
                        save_phong_normals(os.path.join(args.test_path, str(iteration_counter) + "_normals_phong_#.png"), source_n3d)
                        save_depth_from_3d(os.path.join(args.test_path, str(iteration_counter) + "_depth_dn_#.png"), masked_predicted_points3d, 1000.0)
                        save_normals(os.path.join(args.test_path, str(iteration_counter) + "_normals_dn_#.png"), masked_predicted_n3d)
                        save_phong_normals(os.path.join(args.test_path, str(iteration_counter) + "_normals_phong_dn_#.png"), masked_predicted_n3d)

                    print("Testing | Epoch: {} , iteration {}".format(epoch, iteration_counter))
                model.train()

            # visualization (visdom)
            if (iteration_counter + 1) % args.checkpoint_iters <= args.batch_size:
                utils.checkpoint.save_network_state(model, optimizer,epoch,\
                    args.name + "_model_state_epoch_" + str(epoch), args.test_path)
                print("Checkpoint")
            if (iteration_counter + 1) % args.disp_iters <= args.batch_size:
                print("Epoch: {}, iteration: {}\nPhotometric: {}\nDepth: {}\nSurface: {}\nTotal average loss: {}\n\n"\
                    .format(epoch, iteration_counter, running_photo_loss.avg, running_depth_reg_loss.avg,\
                        running_surface_smooth_loss.avg, total_loss.avg))
                #loss plots
                visualizer.append_loss(epoch + 1, iteration_counter, total_loss.avg, "avg")
                visualizer.append_loss(epoch + 1, iteration_counter, running_photo_loss.avg, "photo")
                visualizer.append_loss(epoch + 1, iteration_counter, running_depth_reg_loss.avg, "depth")
                visualizer.append_loss(epoch + 1, iteration_counter, running_surface_smooth_loss.avg, "normal")

                total_loss.reset()
                running_photo_loss.reset()
                running_depth_reg_loss.reset()
                running_surface_smooth_loss.reset()

            if args.visdom_iters > 0 and (iteration_counter + 1) % args.visdom_iters <= args.batch_size:
                for key in processed.keys():
                    splat_imgs = processed[key]["color"]["splatted"]
                    visualizer.show_images(splat_imgs, key + '_splat_img')
                    splat_depths = processed[key]["depth"]["splatted"]
                    visualizer.show_map(splat_depths, key + '_splat_depth')
                visualizer.show_images(photo_l_map, 'photo_loss_imgs')