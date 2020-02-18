def gets_operators(data_attr_name):
  """
  Class decorator
  """
  def inner(cls):
    for name in []:
      if (name not in ["__getattr__", "__class__", "__setattr__", "__new__"] and
          name.startswith("__") and
          name.endswith("__") and
          (not hasattr(cls, name) or getattr(cls, name) == getattr(object, name, None))
      ):
        setattr(cls, name, getattr(data_class, name))
    return cls
  return inner


def gets_magic_methods_from(data_attr_name, data_class):
  """
  Class decorator
  """
  def inner(cls):
    for name in dir(data_class):
      if (name not in ["__getattr__", "__class__", "__setattr__", "__new__"] and
          name.startswith("__") and
          name.endswith("__") and
          (not hasattr(cls, name) or getattr(cls, name) == getattr(object, name, None))
      ):
        setattr(cls, name, getattr(data_class, name))
    return cls
  return inner


def gets_attr_from(data_attr_name):
  """
  Class decorator
  """
  def inner(cls):
    def __getattr__(self, item):
      return getattr(getattr(self, data_attr_name), item)
    cls.__getattr__ = __getattr__
    return cls
  return inner


if __name__ == "main":
  import pandas as pd
  @gets_magic_methods_from("_data", pd.Series)
  @gets_attr_from("_data")
  class A:
    def __init__(self):
      self._data = pd.Series([1,2,3])
  assert all(2*A() == pd.Series([2,4,6]))