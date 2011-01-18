# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.
# This file is compatible with both classic and new-style classes.

from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_pyCombBLAS', [dirname(__file__)])
        except ImportError:
            import _pyCombBLAS
            return _pyCombBLAS
        if fp is not None:
            try:
                _mod = imp.load_module('_pyCombBLAS', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _pyCombBLAS = swig_import_helper()
    del swig_import_helper
else:
    import _pyCombBLAS
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class pySpParMat(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParMat, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParMat, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _pyCombBLAS.new_pySpParMat()
        try: self.this.append(this)
        except: self.this = this
    def getnnz(self): return _pyCombBLAS.pySpParMat_getnnz(self)
    def getnrow(self): return _pyCombBLAS.pySpParMat_getnrow(self)
    def getncol(self): return _pyCombBLAS.pySpParMat_getncol(self)
    def load(self, *args): return _pyCombBLAS.pySpParMat_load(self, *args)
    def GenGraph500Edges(self, *args): return _pyCombBLAS.pySpParMat_GenGraph500Edges(self, *args)
    def GenGraph500Candidates(self, *args): return _pyCombBLAS.pySpParMat_GenGraph500Candidates(self, *args)
    def FindIndsOfColsWithSumGreaterThan(self, *args): return _pyCombBLAS.pySpParMat_FindIndsOfColsWithSumGreaterThan(self, *args)
    def copy(self): return _pyCombBLAS.pySpParMat_copy(self)
    def Apply(self, *args): return _pyCombBLAS.pySpParMat_Apply(self, *args)
    def Prune(self, *args): return _pyCombBLAS.pySpParMat_Prune(self, *args)
    def Reduce(self, *args): return _pyCombBLAS.pySpParMat_Reduce(self, *args)
    def SpMV_PlusTimes(self, *args): return _pyCombBLAS.pySpParMat_SpMV_PlusTimes(self, *args)
    def SpMV_SelMax(self, *args): return _pyCombBLAS.pySpParMat_SpMV_SelMax(self, *args)
    def SpMV_SelMax_inplace(self, *args): return _pyCombBLAS.pySpParMat_SpMV_SelMax_inplace(self, *args)
    __swig_getmethods__["Column"] = lambda x: _pyCombBLAS.pySpParMat_Column
    if _newclass:Column = staticmethod(_pyCombBLAS.pySpParMat_Column)
    __swig_getmethods__["Row"] = lambda x: _pyCombBLAS.pySpParMat_Row
    if _newclass:Row = staticmethod(_pyCombBLAS.pySpParMat_Row)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParMat
    __del__ = lambda self : None;
pySpParMat_swigregister = _pyCombBLAS.pySpParMat_swigregister
pySpParMat_swigregister(pySpParMat)

def pySpParMat_Column():
  return _pyCombBLAS.pySpParMat_Column()
pySpParMat_Column = _pyCombBLAS.pySpParMat_Column

def pySpParMat_Row():
  return _pyCombBLAS.pySpParMat_Row()
pySpParMat_Row = _pyCombBLAS.pySpParMat_Row

class pySpParVec(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParVec, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParVec, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParVec(*args)
        try: self.this.append(this)
        except: self.this = this
    def dense(self): return _pyCombBLAS.pySpParVec_dense(self)
    def getnnz(self): return _pyCombBLAS.pySpParVec_getnnz(self)
    def len(self): return _pyCombBLAS.pySpParVec_len(self)
    def __iadd__(self, *args): return _pyCombBLAS.pySpParVec___iadd__(self, *args)
    def __isub__(self, *args): return _pyCombBLAS.pySpParVec___isub__(self, *args)
    def copy(self): return _pyCombBLAS.pySpParVec_copy(self)
    def SetElement(self, *args): return _pyCombBLAS.pySpParVec_SetElement(self, *args)
    def GetElement(self, *args): return _pyCombBLAS.pySpParVec_GetElement(self, *args)
    def any(self): return _pyCombBLAS.pySpParVec_any(self)
    def all(self): return _pyCombBLAS.pySpParVec_all(self)
    def intersectSize(self, *args): return _pyCombBLAS.pySpParVec_intersectSize(self, *args)
    def printall(self): return _pyCombBLAS.pySpParVec_printall(self)
    def load(self, *args): return _pyCombBLAS.pySpParVec_load(self, *args)
    def Apply(self, *args): return _pyCombBLAS.pySpParVec_Apply(self, *args)
    def SubsRef(self, *args): return _pyCombBLAS.pySpParVec_SubsRef(self, *args)
    def Reduce(self, *args): return _pyCombBLAS.pySpParVec_Reduce(self, *args)
    def Sort(self): return _pyCombBLAS.pySpParVec_Sort(self)
    def setNumToInd(self): return _pyCombBLAS.pySpParVec_setNumToInd(self)
    __swig_getmethods__["zeros"] = lambda x: _pyCombBLAS.pySpParVec_zeros
    if _newclass:zeros = staticmethod(_pyCombBLAS.pySpParVec_zeros)
    __swig_getmethods__["range"] = lambda x: _pyCombBLAS.pySpParVec_range
    if _newclass:range = staticmethod(_pyCombBLAS.pySpParVec_range)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParVec
    __del__ = lambda self : None;
pySpParVec_swigregister = _pyCombBLAS.pySpParVec_swigregister
pySpParVec_swigregister(pySpParVec)

def pySpParVec_zeros(*args):
  return _pyCombBLAS.pySpParVec_zeros(*args)
pySpParVec_zeros = _pyCombBLAS.pySpParVec_zeros

def pySpParVec_range(*args):
  return _pyCombBLAS.pySpParVec_range(*args)
pySpParVec_range = _pyCombBLAS.pySpParVec_range


def EWiseMult(*args):
  return _pyCombBLAS.EWiseMult(*args)
EWiseMult = _pyCombBLAS.EWiseMult

def EWiseMult_inplacefirst(*args):
  return _pyCombBLAS.EWiseMult_inplacefirst(*args)
EWiseMult_inplacefirst = _pyCombBLAS.EWiseMult_inplacefirst
class pyDenseParVec(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pyDenseParVec, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pyDenseParVec, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pyDenseParVec(*args)
        try: self.this.append(this)
        except: self.this = this
    def sparse(self, *args): return _pyCombBLAS.pyDenseParVec_sparse(self, *args)
    def len(self): return _pyCombBLAS.pyDenseParVec_len(self)
    def add(self, *args): return _pyCombBLAS.pyDenseParVec_add(self, *args)
    def __iadd__(self, *args): return _pyCombBLAS.pyDenseParVec___iadd__(self, *args)
    def __isub__(self, *args): return _pyCombBLAS.pyDenseParVec___isub__(self, *args)
    def __add__(self, *args): return _pyCombBLAS.pyDenseParVec___add__(self, *args)
    def __sub__(self, *args): return _pyCombBLAS.pyDenseParVec___sub__(self, *args)
    def copy(self): return _pyCombBLAS.pyDenseParVec_copy(self)
    def SetElement(self, *args): return _pyCombBLAS.pyDenseParVec_SetElement(self, *args)
    def GetElement(self, *args): return _pyCombBLAS.pyDenseParVec_GetElement(self, *args)
    def RandPerm(self): return _pyCombBLAS.pyDenseParVec_RandPerm(self)
    def printall(self): return _pyCombBLAS.pyDenseParVec_printall(self)
    def getnnz(self): return _pyCombBLAS.pyDenseParVec_getnnz(self)
    def getnz(self): return _pyCombBLAS.pyDenseParVec_getnz(self)
    def load(self, *args): return _pyCombBLAS.pyDenseParVec_load(self, *args)
    def Count(self, *args): return _pyCombBLAS.pyDenseParVec_Count(self, *args)
    def Find(self, *args): return _pyCombBLAS.pyDenseParVec_Find(self, *args)
    def FindInds(self, *args): return _pyCombBLAS.pyDenseParVec_FindInds(self, *args)
    def Apply(self, *args): return _pyCombBLAS.pyDenseParVec_Apply(self, *args)
    def ApplyMasked(self, *args): return _pyCombBLAS.pyDenseParVec_ApplyMasked(self, *args)
    def SubsRef(self, *args): return _pyCombBLAS.pyDenseParVec_SubsRef(self, *args)
    __swig_getmethods__["range"] = lambda x: _pyCombBLAS.pyDenseParVec_range
    if _newclass:range = staticmethod(_pyCombBLAS.pyDenseParVec_range)
    __swig_destroy__ = _pyCombBLAS.delete_pyDenseParVec
    __del__ = lambda self : None;
pyDenseParVec_swigregister = _pyCombBLAS.pyDenseParVec_swigregister
pyDenseParVec_swigregister(pyDenseParVec)

def pyDenseParVec_range(*args):
  return _pyCombBLAS.pyDenseParVec_range(*args)
pyDenseParVec_range = _pyCombBLAS.pyDenseParVec_range

class UnaryFunction(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnaryFunction, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UnaryFunction, name)
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_UnaryFunction
    __del__ = lambda self : None;
    def __call__(self, *args): return _pyCombBLAS.UnaryFunction___call__(self, *args)
UnaryFunction_swigregister = _pyCombBLAS.UnaryFunction_swigregister
UnaryFunction_swigregister(UnaryFunction)


def set(*args):
  return _pyCombBLAS.set(*args)
set = _pyCombBLAS.set

def identity():
  return _pyCombBLAS.identity()
identity = _pyCombBLAS.identity

def safemultinv():
  return _pyCombBLAS.safemultinv()
safemultinv = _pyCombBLAS.safemultinv

def abs():
  return _pyCombBLAS.abs()
abs = _pyCombBLAS.abs

def negate():
  return _pyCombBLAS.negate()
negate = _pyCombBLAS.negate

def bitwise_not():
  return _pyCombBLAS.bitwise_not()
bitwise_not = _pyCombBLAS.bitwise_not

def logical_not():
  return _pyCombBLAS.logical_not()
logical_not = _pyCombBLAS.logical_not
class BinaryFunction(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BinaryFunction, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BinaryFunction, name)
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_BinaryFunction
    __del__ = lambda self : None;
    __swig_setmethods__["commutable"] = _pyCombBLAS.BinaryFunction_commutable_set
    __swig_getmethods__["commutable"] = _pyCombBLAS.BinaryFunction_commutable_get
    if _newclass:commutable = _swig_property(_pyCombBLAS.BinaryFunction_commutable_get, _pyCombBLAS.BinaryFunction_commutable_set)
    __swig_setmethods__["associative"] = _pyCombBLAS.BinaryFunction_associative_set
    __swig_getmethods__["associative"] = _pyCombBLAS.BinaryFunction_associative_get
    if _newclass:associative = _swig_property(_pyCombBLAS.BinaryFunction_associative_get, _pyCombBLAS.BinaryFunction_associative_set)
    def __call__(self, *args): return _pyCombBLAS.BinaryFunction___call__(self, *args)
BinaryFunction_swigregister = _pyCombBLAS.BinaryFunction_swigregister
BinaryFunction_swigregister(BinaryFunction)


def plus():
  return _pyCombBLAS.plus()
plus = _pyCombBLAS.plus

def minus():
  return _pyCombBLAS.minus()
minus = _pyCombBLAS.minus

def multiplies():
  return _pyCombBLAS.multiplies()
multiplies = _pyCombBLAS.multiplies

def divides():
  return _pyCombBLAS.divides()
divides = _pyCombBLAS.divides

def modulus():
  return _pyCombBLAS.modulus()
modulus = _pyCombBLAS.modulus

def max():
  return _pyCombBLAS.max()
max = _pyCombBLAS.max

def min():
  return _pyCombBLAS.min()
min = _pyCombBLAS.min

def bitwise_and():
  return _pyCombBLAS.bitwise_and()
bitwise_and = _pyCombBLAS.bitwise_and

def bitwise_or():
  return _pyCombBLAS.bitwise_or()
bitwise_or = _pyCombBLAS.bitwise_or

def bitwise_xor():
  return _pyCombBLAS.bitwise_xor()
bitwise_xor = _pyCombBLAS.bitwise_xor

def logical_and():
  return _pyCombBLAS.logical_and()
logical_and = _pyCombBLAS.logical_and

def logical_or():
  return _pyCombBLAS.logical_or()
logical_or = _pyCombBLAS.logical_or

def logical_xor():
  return _pyCombBLAS.logical_xor()
logical_xor = _pyCombBLAS.logical_xor

def equal_to():
  return _pyCombBLAS.equal_to()
equal_to = _pyCombBLAS.equal_to

def not_equal_to():
  return _pyCombBLAS.not_equal_to()
not_equal_to = _pyCombBLAS.not_equal_to

def greater():
  return _pyCombBLAS.greater()
greater = _pyCombBLAS.greater

def less():
  return _pyCombBLAS.less()
less = _pyCombBLAS.less

def greater_equal():
  return _pyCombBLAS.greater_equal()
greater_equal = _pyCombBLAS.greater_equal

def less_equal():
  return _pyCombBLAS.less_equal()
less_equal = _pyCombBLAS.less_equal

def bind1st(*args):
  return _pyCombBLAS.bind1st(*args)
bind1st = _pyCombBLAS.bind1st

def bind2nd(*args):
  return _pyCombBLAS.bind2nd(*args)
bind2nd = _pyCombBLAS.bind2nd

def compose1(*args):
  return _pyCombBLAS.compose1(*args)
compose1 = _pyCombBLAS.compose1

def compose2(*args):
  return _pyCombBLAS.compose2(*args)
compose2 = _pyCombBLAS.compose2

def not1(*args):
  return _pyCombBLAS.not1(*args)
not1 = _pyCombBLAS.not1

def not2(*args):
  return _pyCombBLAS.not2(*args)
not2 = _pyCombBLAS.not2

def finalize():
  return _pyCombBLAS.finalize()
finalize = _pyCombBLAS.finalize

def root():
  return _pyCombBLAS.root()
root = _pyCombBLAS.root


