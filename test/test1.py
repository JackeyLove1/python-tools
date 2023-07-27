from enum import Enum
import random

class CPUFault:
    def run(self):
        print("Running CPU fault...")

class MemoryFault:
    def run(self):
        print("Running memory fault...")

class DiskFault:
    def run(self):
        print("Running disk fault...")

class NetworkFault:
    def run(self):
        print("Running network fault...")

class Fault(Enum):
    CPU = CPUFault()
    Memory = MemoryFault()
    Disk = DiskFault()
    Network = NetworkFault()

# randomly choose a fault
chosen_fault = random.choice(list(Fault))
print(list(Fault))
# run the chosen fault
chosen_fault.value.run()