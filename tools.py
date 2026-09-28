import time
import os
import numpy as np
import re
import matplotlib
import random
import turtle
import pandas as pd
import json
import scipy
import csv
import ast
import seaborn
import math
import datetime
import subprocess
import shutil
import pathlib
import sys
import requests
import httpx
import bs4

toolInfo = """
-- LIBRARIES --
time
np -> numpy
os
re
matplotlib
random
turtle
pd -> pandas
json
scipy
csv
ast
seaborn
math
datetime
subprocess
shutil
pathlib
sys
bs4 -> beautiful soup 4
requests
httpx

-- FUNCTIONS --
afterQuestionAsked -> call after you ask for a question. you should always do this after a question.
"""

state = {
	"isQuestionAsked": False,
}

def afterQuestionAsked():
	state["isQuestionAsked"] = True

def executeInVenv(code:str, timeout:int = 30) -> str:
	wrapperCode = f"""import sys
sys.path = {sys.path!r}
from tools import *
{code}"""

	try:
		result = subprocess.run(
			[sys.executable, "-c", wrapperCode],
			capture_output=True,
			text=True,
			timeout=timeout
		)

		if result.returncode == 0:
			return result.stdout or "success"
		else:
			return f"error:\n{result.stderr}"

	except subprocess.TimeoutExpired:
		return f"execution timed out after {timeout} seconds."
	except Exception as e:
		return f"system execution error: {str(e)}"

def processText(txt):
	state["isQuestionAsked"] = False
	inCurly = re.findall(r"\{([^}]+)\}", txt)
	newTxt = re.sub(r"\{[^}]*\}", "", txt).strip()

	for x in inCurly:
		print("--- PYTHON CODE: " + x)
		exec(x)

	return newTxt
