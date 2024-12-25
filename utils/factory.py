def get_model(model_name, args):
    name = model_name.lower()
    if name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "wa":
        from models.wa import WA
        return WA(args)
    elif name == "podnet":
        from models.podnet import PODNet
        return PODNet(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "foster":
        from models.foster import FOSTER
        return FOSTER(args)
    elif name == "memo":
        from models.memo import MEMO
        return MEMO(args)
    else:
        assert 0
