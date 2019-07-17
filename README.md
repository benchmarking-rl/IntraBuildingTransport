#ElevatorSimulator

ElevatorSimulator是一个电梯调度模拟环境

<img src="demo_image.gif" width="400"/>



## 下载

可以通过pip下载：

```python
pip install liftsim
```



## 基本接口

类似gym，ElevatorSimulator提供了三个基本接口：

- reset(self)：重置环境，返回observation。
- step(self, action)：根据action调整环境，返回observation，reward，done，info。
- render(self)：显示一个timestep内的环境。

## 使用

### 配置大楼

你需要通过一个配置文件来配置大楼各项参数，如楼层数、层高、电梯数、乘客产生频率、调度的timestep和log种类。可以参考[示例][config]。

- RunningTimeStep：即timestep，调用一次step( )电梯将移动的时长。不宜过大，0.10~0.50即可。
- LogLevel：有Debug、Notice、Warning三种（注意大小写）。
- Lognorm（可选）：输出log信息到该文件。
- Logerr（可选）：输出error信息到该文件。
- ParticleNumber/GenerationInterval：调节乘客产生频率。

### 调控电梯

通过dispatcher来调度电梯。dispatcher需继承[DispatcherBase][dispatch]。

- link_mansion(self, mansion_attr)：传入大楼配置到dispatcher。通过mansion_attr获取如大楼内电梯数等信息。

- policy(self, state)：你需实现此方法，接收MansionState，返回调度的action。

  - MansionState：namedtuple，包括ElevatorStates（各个电梯情况）、RequiringUpwardFloors（list，保存有乘客等待上行的楼层）、RequiringDownwardFloors（list，保存有乘客等待下行的楼层）。

  - ElevatorState：namedtuple，包括电梯当前楼层、速度、电梯调配到的楼层等等。

  - action：policy(self, state)需返回一个ElevatorAction的list。

  - ElevatorAction：namedtuple，包括TargetFloor（分配的楼层）及DirectionIndicator（分配的方向）。

    TargetFloor可为-1~最高楼层。-1表示不改变电梯当前分配到的楼层；0表示立即减速停下（当电梯为空时才有效，否则继续运行）；其余楼层即正常分配楼层。

    DirectionIndicator为-1（向下）、0（无方向）、1（向上）。

  - **关于MansionState、ElevatorState、ElevatorAction，详见[mansion/utils][utils]。**

  - [baseline/utils][baseline/utils]中有处理MansionState及ElevatorStates的函数供参考。

### 运行逻辑

电梯负责处理电梯内乘客按下的楼层（target_floot），依次停靠。Dispatcher负责调配电梯到有人等待的楼层去接乘客（dispatch_target）以及分配接到人后的方向（dispatch_target_direction）。

电梯若无法在dispatch_target停下，则忽略dispatch_target。若希望电梯在某一层停下，则需要在电梯停在该楼层之前，保持dispatch_target一直为该楼层，或返回ElevatorAction中TargetFloor为-1。

dispatch_target_direction负责指示接到乘客后电梯行驶方向，当电梯内无人，电梯静止且无方向时有效。

## 示例

我们提供了一个基于规则的[电梯调度算法][demo]，可用于参考电梯环境的使用。



[config]: https://github.com/benchmarking-rl/IntraBuildingTransport/blob/master/config.ini
[dispatch]: https://github.com/benchmarking-rl/IntraBuildingTransport/blob/master/liftsim/dispatcher_base.py
[utils]: https://github.com/benchmarking-rl/IntraBuildingTransport/blob/master/liftsim/mansion/utils.py
[demo]: https://github.com/benchmarking-rl/IntraBuildingTransport/blob/master/demo.py
[baseline/utils]: https://github.com/benchmarking-rl/IntraBuildingTransport/blob/master/baseline/utils.py