import sys
import time
for x in range (0,5):  
    b = "Loading" + "." * x
    print (b, end="\r")
    time.sleep(1)

from .func import *
from .plots import *
from .classes import *

print("Package: Resources loaded. ☑")
