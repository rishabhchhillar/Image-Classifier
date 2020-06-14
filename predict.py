#Imports
import json
from utils import *

#Define main function
def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_checkpoint(args.checkpoint)
    
    image_t = process_image(args.image)
    
    device = check_gpu(args.gpu)
    
    top_ps, top_classes, top_flowers = predict(image_t, model, device, cat_to_name, args.topk)
    
    print_probs(top_flowers, top_ps)
    
if __name__ == '__main__':
    main()
    

    
    
    
    
    
    