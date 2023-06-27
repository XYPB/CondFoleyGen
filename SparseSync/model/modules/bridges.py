import torch


class BridgeBase(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.bridge = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.bridge(x)
            return x
        except TypeError as e:
            raise TypeError('The class cant be called on its own. Please, use a class that inherits it', e)


class ConvBridgeBase(BridgeBase):

    def __init__(self, block, **kwargs) -> None:
        super().__init__()
        self.bridge = torch.nn.Sequential(
            block(**kwargs),
            torch.nn.GELU(),
        )


class AvgPoolBridgeBase(BridgeBase):

    def __init__(self, block, **kwargs) -> None:
        super().__init__()
        # a list [1, 2, 3] specified in omegaconfig, has type ListConf which is not accepted by pytorch
        # interestingly, conv layers don't care
        for k in ['kernel_size', 'stride']:
            kwargs[k] = list(kwargs[k])
        self.bridge = block(**kwargs)


class ConvBridgeAudio(ConvBridgeBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(block=torch.nn.Conv2d, **kwargs)


class ConvBridgeVisual(ConvBridgeBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(block=torch.nn.Conv3d, **kwargs)


class AvgPoolBridgeVisual(AvgPoolBridgeBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(block=torch.nn.AvgPool3d, **kwargs)


class AvgPoolBridgeAudio(AvgPoolBridgeBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(block=torch.nn.AvgPool2d, **kwargs)


class DoNothingBridge(BridgeBase):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.bridge = torch.nn.Identity(**kwargs)


class AppendZerosToHidden(BridgeBase):

    def __init__(self, target_hidden_size, dim) -> None:
        super().__init__()
        self.target_hidden_size = target_hidden_size
        self.dim = dim

    def forward(self, x):
        d_res = self.target_hidden_size - x.shape[self.dim]
        # going to insert the new dimension into the x.shape output
        shape_target = list(x.shape[:self.dim]) + [d_res] + list(x.shape[self.dim+1:])
        # creating the zeros to append to x
        zeros = torch.zeros(shape_target).to(x)
        x = torch.cat([x, zeros], self.dim)
        return x



if __name__ == '__main__':
    v = torch.rand(2, 50, 512, 7, 7)
    a = torch.rand(2, 512, 9, 27)

    in_channels = 512
    out_channels = 512
    kernel_size_v = [1, 7, 7]
    kernel_size_a = [9, 1]
    stride = 1
    bias = True

    conv_bridge_a = ConvBridgeAudio(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size_a, stride=stride, bias=bias)
    conv_bridge_v = ConvBridgeVisual(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size_v, stride=stride, bias=bias)
    avg_bridge_a = AvgPoolBridgeAudio(kernel_size=kernel_size_a)
    avg_bridge_v = AvgPoolBridgeVisual(kernel_size=kernel_size_v)
    i_bridge_a = DoNothingBridge(some_arg=123)
    i_bridge_v = DoNothingBridge(some_arg=123)
    h_bridge_a = AppendZerosToHidden(target_hidden_size=1024, dim=1)
    h_bridge_v = AppendZerosToHidden(target_hidden_size=1024, dim=1)

    print('v', v.shape)
    print('conv_v(v)', conv_bridge_v(v.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).shape)
    print()

    print('a', a.shape)
    print('conv_a(a)', conv_bridge_a(a).shape)
    print()

    print('v', v.shape)
    print('avg3d(v)', avg_bridge_v(v.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).shape)
    print()

    print('a', a.shape)
    print('avg2d(a)', avg_bridge_a(a).shape)
    print()

    print('v', v.shape)
    print('i(v)', i_bridge_v(v.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).shape)
    print()

    print('a', a.shape)
    print('i(a)', i_bridge_a(a).shape)
    print()

    print('v', v.shape)
    print('h(v)', h_bridge_v(v.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).shape)
    print()

    print('a', a.shape)
    print('h(a)', h_bridge_a(a).shape)
    print()
