{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "29f7c24d",
   "metadata": {
    "_cell_guid": "4edd95b8-3ba9-4ba8-a66e-0da9ab96c7e1",
    "_uuid": "b88b15b6-da12-4ba3-8d4c-e6d58b6c7d10",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2024-11-03T02:07:40.253848Z",
     "iopub.status.busy": "2024-11-03T02:07:40.253364Z",
     "iopub.status.idle": "2024-11-03T02:07:41.306587Z",
     "shell.execute_reply": "2024-11-03T02:07:41.304998Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.060256,
     "end_time": "2024-11-03T02:07:41.309388",
     "exception": false,
     "start_time": "2024-11-03T02:07:40.249132",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "106.33333333333333\n",
      "11.0\n",
      "The object is a sphere\n"
     ]
    }
   ],
   "source": [
    "# start of code\n",
    "\n",
    "# this is by NO MEANS finished, it is just the starting code for calculating size and color stuff\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import random\n",
    "\n",
    "x = random_int = random.randint(0, 255)\n",
    "y = random_int = random.randint(0, 255)\n",
    "z = random_int = random.randint(0, 255)\n",
    "RGB = [x, y, z]\n",
    "\n",
    "a = random_int = random.randint(0, 30)\n",
    "b = random_int = random.randint(0, 30)\n",
    "c = random_int = random.randint(0, 30)\n",
    "size = [a, b, c]\n",
    "\n",
    "average = (x + y + z) / len(RGB)\n",
    "print(average)\n",
    "\n",
    "avg_size = (size[0] + size[1] + size[2]) / len(size)\n",
    "print(avg_size)\n",
    "\n",
    "if average > 100:\n",
    "    if avg_size > 20:\n",
    "        print(\"Still have a defeciency\")\n",
    "    elif 10 < avg_size < 20:\n",
    "        print(\"The object is a sphere\")\n",
    "    else:\n",
    "        print(\"The object is a cube\")\n",
    "if average < 100:\n",
    "    if avg_size > 20:\n",
    "        print(\"The object is a cube\")\n",
    "    elif 10 < avg_size < 20:\n",
    "        print(\"The object is a sphere\")\n",
    "    else:\n",
    "        print(\"Still have a defeciency\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bceedacc",
   "metadata": {
    "papermill": {
     "duration": 0.002096,
     "end_time": "2024-11-03T02:07:41.314012",
     "exception": false,
     "start_time": "2024-11-03T02:07:41.311916",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [],
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 5.179496,
   "end_time": "2024-11-03T02:07:41.939914",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-11-03T02:07:36.760418",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
