import pandas as pd
#Pandas Series 1-d labeled array
ser = pd.Series(data=[100,200,300], index=['tom','bob','nancy'])
ser = pd.Series([100, 'foo', 300, 'bar', 500], ['tom', 'bob', 'nancy', 'dan', 'eric'])
print(ser)
print(ser.index)
print(ser.loc[['nancy','bob']])
print(ser[[2,3]])
print('bob' in ser)

#dictionary
d = {'one' : pd.Series([100., 200., 300.], index=['apple', 'ball', 'clock']),
     'two' : pd.Series([111., 222., 333., 4444.], index=['apple', 'ball', 'cerill', 'dancy'])}
df = pd.DataFrame(d)
print(df)
print(df.index)
print(df.columns)
print(pd.DataFrame(d, index=['dancy', 'ball', 'apple']))
print(pd.DataFrame(d, index=['dancy', 'ball', 'apple'], columns=['two', 'five']))
# data array of 2d,
# alex': 1, 'joe': 2 in 1
#'ema': 5, 'dora': 10, 'alice': 20 in 2
data = [{'alex': 1, 'joe': 2}, {'ema': 5, 'dora': 10, 'alice': 20}]
pd.DataFrame(data)
#set index name
pd.DataFrame(data, index=['a','b'])
#selecct some of the elements from the dic as cols
pd.DataFrame(data, columns=['joe','dora'])
#set index name and select some elements
pd.DataFrame(pd.DataFrame(data,index=['a','b']), columns=['joe', 'dora','alice'])

print(df)
print(df['one'])
df['three'] = df['one'] * df['two']
print(df)
df['flag'] = df['one'] > 250
print(df)
three = df.pop("three")
print(df)
del df['two']
df.insert(2, 'copy_of_one', df['one'])


