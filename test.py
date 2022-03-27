import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataloader import GetLoader
from torchvision import datasets


def test(dataset_name, epoch):

    image_root = os.path.join('data', dataset_name).replace("\\","/")

    cuda = True
    cudnn.benchmark = True
    batch_size = 8
    image_size = 227
    alpha = 0

    # load data

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_label_list = os.path.join('data', 'train_labels.csv').replace("\\","/")
    test_label_list = os.path.join('data', 'test_labels.csv').replace("\\","/")

    dataset_source = GetLoader(
        data_root=os.path.join(image_root, 'train_set').replace("\\","/"),
        data_list=train_label_list,
        transform=img_transform_target
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True)

    dataset_target = GetLoader(
        data_root=os.path.join(image_root, 'test_set').replace("\\","/"),
        data_list=test_label_list,
        transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True
    )

    """ training """

    my_net = torch.load(os.path.join('output', 'model_epoch_' + str(epoch) + '.pth').replace("\\","/"))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader_source)
    data_target_iter = iter(dataloader_source)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))