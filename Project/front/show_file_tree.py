# 폴더를 선택하면 그 폴더의 디렉터리 구조(file_tree)를 보여줌
import os
import sys
import tkinter
from tkinter import ttk
from  tkinter import filedialog

def fill_tree(treeview, node):
    if treeview.set(node, "type") != 'directory':
        return

    path = treeview.set(node, "fullpath")
    # Delete the possibly 'dummy' node present.
    treeview.delete(*treeview.get_children(node))

    parent = treeview.parent(node)
    for p in os.listdir(path):
        p = os.path.join(path, p)
        ptype = None
        if os.path.isdir(p):
            ptype = 'directory'

        fname = os.path.split(p)[1]
        oid = treeview.insert(node, 'end', text=fname, values=[p, ptype])
        if ptype == 'directory':
            treeview.insert(oid, 0, text='dummy')

def update_tree(event):
    treeview = event.widget
    fill_tree(treeview, treeview.focus())

def create_root(treeview, startpath):
    dfpath = os.path.abspath(startpath)
    node = treeview.insert('', 'end', text=dfpath,
            values=[dfpath, "directory"], open=True)
    fill_tree(treeview, node)


root = tkinter.Tk()
root.filename =  filedialog.askdirectory()
path = os.path.realpath(root.filename)
treeview = ttk.Treeview(columns=("fullpath", "type"), displaycolumns='')
treeview.pack(fill='both', expand=True)
print(sys.argv)
create_root(treeview, path)
treeview.bind('<<TreeviewOpen>>', update_tree)

root.mainloop()