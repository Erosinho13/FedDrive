from .bisenetv2 import BiSeNetV2

def make_model(args):
    if args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes, output_aux=args.output_aux, pretrained=args.pretrained)
    else:
        raise NotImplementedError("Specify a correct model.")

    return model
