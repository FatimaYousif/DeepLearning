import time
import torch.optim
from task1_loading import MNISTMetricDataset
from torch.utils.data import DataLoader
from task2_model import SimpleMetricEmbedding
from identity_model import IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


def main1():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "Lab1/FCNNs/mnist"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=False,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1
    )

    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()
        train_loss = train(model, optimizer, train_loader, device, name="model_"+str(epoch))
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")
    

    print("Evaluating on valid set model...")
    acc1 = evaluate(model, representations, traineval_loader, device)
    print(f"Valid Top1 Acc: {round(acc1 * 100, 2)}%")

    identity_model = IdentityModel()
    image_representations = compute_representations(identity_model, traineval_loader, identities_count=10, emb_size=784)
    print("Evaluating on valid set Identity model...")
    acc1 = evaluate(identity_model, image_representations, traineval_loader, device)
    print(f"Valid Top1 Acc: {round(acc1 * 100, 2)}%")

def main2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "Lab1/FCNNs/mnist"
    train_loader = DataLoader(
        MNISTMetricDataset(mnist_download_root, split='train'),
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    
    # removed class 0
    train_loader_remove = DataLoader(
        MNISTMetricDataset(mnist_download_root, split='train', remove_class=0),
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    traineval_loader = DataLoader(
        MNISTMetricDataset(mnist_download_root, split='traineval'),
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )

    test_loader = DataLoader(
        MNISTMetricDataset(mnist_download_root, split='test'),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    num_classes = 10
    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()
        train_loss = train(model, optimizer, train_loader_remove, device, name="reduced_model_"+str(epoch))
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1 - t0) / 10 ** 9:.1f}")
    

if __name__ == '__main__':
    # main1()            # w trained + identity (baseline) model  
    main2()             # w/o class 0  -- only trained model  