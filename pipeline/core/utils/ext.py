import textwrap
import builtins

def prettyprint(st,level=1,n=80,tag=""):
    l = "\n" + tag + "".join(["\t"]*level)
    return l+l.join(textwrap.wrap(st, n))

class prefixprint:
    def __init__(self,level=1,n=80,tag=""):
        self.level = level
        self.n = n
        self.tag = tag
    def printfunction(self, *objs, **kwargs):
        sep = kwargs.get('sep', ' ')
        l = "\n" + self.tag + "".join(["\t"]*self.level)
        strs = l + l.join(textwrap.wrap(str(sep.join(objs)), self.n))
        builtins.print(strs, **kwargs)
        
    def __enter__(self):
        return self.printfunction
    def __exit__(self ,type, value, traceback):
        return False

