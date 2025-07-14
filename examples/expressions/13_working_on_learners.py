"""
.. _example_working_on_learners:

.. |Learner| replace::
    :class:`~skrub.SkrubPipeline`


Extending skrub learners after creation
=================================================

In this example, we show how to modify a skrub |Learner| after it has been created.

We start by creating a simple DataOps plan on a single dataset, then we show how to
extend it after it has already been saved to a file.
"""

# %%
import skrub
from skrub.datasets import fetch_employee_salaries

# %%
# We start by fetching the dataset, then by selecting only the first 80% of the
# data as "training data": the rest will be used later as "unseen data" to test
# our learner on.
data = fetch_employee_salaries()

all_data = data.employee_salaries
train_data = all_data[: int(0.8 * len(all_data))]
unseen_data = all_data[int(0.8 * len(all_data)) :]

# %%
# Now, we create a simple DataOps plan that will be trained on the training data,
# and then saved to a file.
# We begin by creating the variables, and by sampling them to iterate quicker.
# You may find more detail on subsampling in the
# :ref:`subsampling example <example_subsampling>`.

train = skrub.var("data", train_data).skb.subsample()
X = train.skb.drop("current_annual_salary").skb.mark_as_X()
y = train.skb.select("current_annual_salary").skb.mark_as_y()
X
# %%
from skrub import TableVectorizer

X_vec = X.skb.apply(TableVectorizer()).skb.set_name("X_vec")
X_vec
# %%
from sklearn.ensemble import HistGradientBoostingRegressor

predictions = X_vec.skb.apply(
    HistGradientBoostingRegressor(
        random_state=42, learning_rate=skrub.choose_float(0.01, 1, log=True, name="lr")
    ),
    y=y,
).skb.set_name("predictor")
predictions
# %%
predictions.skb.cross_validate()
# %%
search = predictions.skb.get_randomized_search(n_iter=10, random_state=42, fitted=True)
learner = search.best_pipeline_

# %%
import cloudpickle

with open("employee_learner.pkl", "wb") as f:
    cloudpickle.dump(learner, f)


# %%
with open("employee_learner.pkl", "rb") as f:
    loaded_learner = cloudpickle.load(f)
# %%
# Now we can use the loaded learner to make predictions on unseen data.
loaded_learner.predict({"data": unseen_data})
# %%
# We can retrieve the DataOps plan used to build the learner using ``.expr``:
loaded_expr = loaded_learner.expr
cloned_expr = loaded_expr.skb.clone()
cloned_expr
# %%
with open("expr.pkl", "wb") as f:
    cloudpickle.dump(cloned_expr, f)
# %%
cloned_expr
# %%
skrub.cross_validate(cloned_expr.skb.get_pipeline(), {"data": train_data})
# %%
