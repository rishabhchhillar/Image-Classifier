#Imports
from torch import nn, optim
from funcs import *
from workspace_utils import active_session

#Define main function
def main():
    
    args = arg_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transforms(train_dir)
    valid_data = valid_transforms(valid_dir)
    test_data = test_transforms(test_dir)
    
    trainloader = train_loader(train_data)
    validloader = valid_loader(valid_data)
    testloader = test_loader(test_data)
    
    model = load_model(args.arch)
    
    model.classifier = build_classifier(model, args.hidden_units, args.dropout)
    
    device = check_gpu(args.gpu)
    
    model.to(device)
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    with active_session():
        trained_model = train_network(model, criterion, optimizer, args.epochs, device, trainloader, validloader)
    
    test_model(model, criterion, testloader, device)
    
    save_checkpoint(model, args.arch, args.save_dir, train_data)
    
if __name__ == '__main__':
    main()
