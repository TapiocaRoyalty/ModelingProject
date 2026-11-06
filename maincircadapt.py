
import matplotlib.pyplot as plt
from circadapt import VanOsta2024

model = VanOsta2024()
model.run(stable=True)
model.plot()
plt.show()