import jpype
import asposecells

jpype.startJVM()
from asposecells.api import Workbook

workbook = Workbook("result.xlsx")
workbook.Save("Output.csv")
jpype.shutdownJVM()