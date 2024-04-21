from data_provider.data_source import DatasetWaveHeight
from torch.utils.data import DataLoader


def data_provider(args, flag):
    data_set = DatasetWaveHeight(
        model=args.model,
        station_id=args.station_id,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        years=args.years,
        flag=flag,
        data_rootpath= args.data_rootpath
    )

    if flag == 'train':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = data_set.__len__()

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )
    return data_set, data_loader
