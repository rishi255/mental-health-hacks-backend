Бљ"
єП
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
≠
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8Н§
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

: *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
»
5token_and_position_embedding_2/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ні *F
shared_name75token_and_position_embedding_2/embedding_4/embeddings
Ѕ
Itoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_4/embeddings* 
_output_shapes
:
Ні *
dtype0
«
5token_and_position_embedding_2/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и *F
shared_name75token_and_position_embedding_2/embedding_5/embeddings
ј
Itoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_5/embeddings*
_output_shapes
:	и *
dtype0
ќ
7transformer_block_2/multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *H
shared_name97transformer_block_2/multi_head_attention_2/query/kernel
«
Ktransformer_block_2/multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_2/multi_head_attention_2/query/kernel*"
_output_shapes
: 
 *
dtype0
∆
5transformer_block_2/multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *F
shared_name75transformer_block_2/multi_head_attention_2/query/bias
њ
Itransformer_block_2/multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_2/multi_head_attention_2/query/bias*
_output_shapes

:
 *
dtype0
 
5transformer_block_2/multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *F
shared_name75transformer_block_2/multi_head_attention_2/key/kernel
√
Itransformer_block_2/multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_2/multi_head_attention_2/key/kernel*"
_output_shapes
: 
 *
dtype0
¬
3transformer_block_2/multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *D
shared_name53transformer_block_2/multi_head_attention_2/key/bias
ї
Gtransformer_block_2/multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_2/multi_head_attention_2/key/bias*
_output_shapes

:
 *
dtype0
ќ
7transformer_block_2/multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *H
shared_name97transformer_block_2/multi_head_attention_2/value/kernel
«
Ktransformer_block_2/multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_2/multi_head_attention_2/value/kernel*"
_output_shapes
: 
 *
dtype0
∆
5transformer_block_2/multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *F
shared_name75transformer_block_2/multi_head_attention_2/value/bias
њ
Itransformer_block_2/multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_2/multi_head_attention_2/value/bias*
_output_shapes

:
 *
dtype0
д
Btransformer_block_2/multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *S
shared_nameDBtransformer_block_2/multi_head_attention_2/attention_output/kernel
Ё
Vtransformer_block_2/multi_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_2/multi_head_attention_2/attention_output/kernel*"
_output_shapes
:
  *
dtype0
Ў
@transformer_block_2/multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_2/multi_head_attention_2/attention_output/bias
—
Ttransformer_block_2/multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_2/multi_head_attention_2/attention_output/bias*
_output_shapes
: *
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

: d*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:d*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d *
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:d *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
: *
dtype0
ґ
/transformer_block_2/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_2/layer_normalization_4/gamma
ѓ
Ctransformer_block_2/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_2/layer_normalization_4/gamma*
_output_shapes
: *
dtype0
і
.transformer_block_2/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_2/layer_normalization_4/beta
≠
Btransformer_block_2/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp.transformer_block_2/layer_normalization_4/beta*
_output_shapes
: *
dtype0
ґ
/transformer_block_2/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_2/layer_normalization_5/gamma
ѓ
Ctransformer_block_2/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_2/layer_normalization_5/gamma*
_output_shapes
: *
dtype0
і
.transformer_block_2/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_2/layer_normalization_5/beta
≠
Btransformer_block_2/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp.transformer_block_2/layer_normalization_5/beta*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
И
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_10/kernel/m
Б
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

: *
dtype0
А
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/m
Б
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
÷
<Adam/token_and_position_embedding_2/embedding_4/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ні *M
shared_name><Adam/token_and_position_embedding_2/embedding_4/embeddings/m
ѕ
PAdam/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_4/embeddings/m* 
_output_shapes
:
Ні *
dtype0
’
<Adam/token_and_position_embedding_2/embedding_5/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и *M
shared_name><Adam/token_and_position_embedding_2/embedding_5/embeddings/m
ќ
PAdam/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_5/embeddings/m*
_output_shapes
:	и *
dtype0
№
>Adam/transformer_block_2/multi_head_attention_2/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m
’
RAdam/transformer_block_2/multi_head_attention_2/query/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m*"
_output_shapes
: 
 *
dtype0
‘
<Adam/transformer_block_2/multi_head_attention_2/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/query/bias/m
Ќ
PAdam/transformer_block_2/multi_head_attention_2/query/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/query/bias/m*
_output_shapes

:
 *
dtype0
Ў
<Adam/transformer_block_2/multi_head_attention_2/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/key/kernel/m
—
PAdam/transformer_block_2/multi_head_attention_2/key/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/key/kernel/m*"
_output_shapes
: 
 *
dtype0
–
:Adam/transformer_block_2/multi_head_attention_2/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *K
shared_name<:Adam/transformer_block_2/multi_head_attention_2/key/bias/m
…
NAdam/transformer_block_2/multi_head_attention_2/key/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_2/multi_head_attention_2/key/bias/m*
_output_shapes

:
 *
dtype0
№
>Adam/transformer_block_2/multi_head_attention_2/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m
’
RAdam/transformer_block_2/multi_head_attention_2/value/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m*"
_output_shapes
: 
 *
dtype0
‘
<Adam/transformer_block_2/multi_head_attention_2/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/value/bias/m
Ќ
PAdam/transformer_block_2/multi_head_attention_2/value/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/value/bias/m*
_output_shapes

:
 *
dtype0
т
IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *Z
shared_nameKIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m
л
]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m*"
_output_shapes
:
  *
dtype0
ж
GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/m
я
[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

: d*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:d*
dtype0
Ж
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d *&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:d *
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
: *
dtype0
ƒ
6Adam/transformer_block_2/layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_4/gamma/m
љ
JAdam/transformer_block_2/layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_4/gamma/m*
_output_shapes
: *
dtype0
¬
5Adam/transformer_block_2/layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_4/beta/m
ї
IAdam/transformer_block_2/layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_4/beta/m*
_output_shapes
: *
dtype0
ƒ
6Adam/transformer_block_2/layer_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_5/gamma/m
љ
JAdam/transformer_block_2/layer_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_5/gamma/m*
_output_shapes
: *
dtype0
¬
5Adam/transformer_block_2/layer_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_5/beta/m
ї
IAdam/transformer_block_2/layer_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_5/beta/m*
_output_shapes
: *
dtype0
И
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_10/kernel/v
Б
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

: *
dtype0
А
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/v
Б
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
÷
<Adam/token_and_position_embedding_2/embedding_4/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ні *M
shared_name><Adam/token_and_position_embedding_2/embedding_4/embeddings/v
ѕ
PAdam/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_4/embeddings/v* 
_output_shapes
:
Ні *
dtype0
’
<Adam/token_and_position_embedding_2/embedding_5/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и *M
shared_name><Adam/token_and_position_embedding_2/embedding_5/embeddings/v
ќ
PAdam/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_5/embeddings/v*
_output_shapes
:	и *
dtype0
№
>Adam/transformer_block_2/multi_head_attention_2/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v
’
RAdam/transformer_block_2/multi_head_attention_2/query/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v*"
_output_shapes
: 
 *
dtype0
‘
<Adam/transformer_block_2/multi_head_attention_2/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/query/bias/v
Ќ
PAdam/transformer_block_2/multi_head_attention_2/query/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/query/bias/v*
_output_shapes

:
 *
dtype0
Ў
<Adam/transformer_block_2/multi_head_attention_2/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/key/kernel/v
—
PAdam/transformer_block_2/multi_head_attention_2/key/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/key/kernel/v*"
_output_shapes
: 
 *
dtype0
–
:Adam/transformer_block_2/multi_head_attention_2/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *K
shared_name<:Adam/transformer_block_2/multi_head_attention_2/key/bias/v
…
NAdam/transformer_block_2/multi_head_attention_2/key/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_2/multi_head_attention_2/key/bias/v*
_output_shapes

:
 *
dtype0
№
>Adam/transformer_block_2/multi_head_attention_2/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
 *O
shared_name@>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v
’
RAdam/transformer_block_2/multi_head_attention_2/value/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v*"
_output_shapes
: 
 *
dtype0
‘
<Adam/transformer_block_2/multi_head_attention_2/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *M
shared_name><Adam/transformer_block_2/multi_head_attention_2/value/bias/v
Ќ
PAdam/transformer_block_2/multi_head_attention_2/value/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_2/multi_head_attention_2/value/bias/v*
_output_shapes

:
 *
dtype0
т
IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *Z
shared_nameKIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v
л
]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v*"
_output_shapes
:
  *
dtype0
ж
GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/v
я
[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

: d*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:d*
dtype0
Ж
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d *&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:d *
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
: *
dtype0
ƒ
6Adam/transformer_block_2/layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_4/gamma/v
љ
JAdam/transformer_block_2/layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_4/gamma/v*
_output_shapes
: *
dtype0
¬
5Adam/transformer_block_2/layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_4/beta/v
ї
IAdam/transformer_block_2/layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_4/beta/v*
_output_shapes
: *
dtype0
ƒ
6Adam/transformer_block_2/layer_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_5/gamma/v
љ
JAdam/transformer_block_2/layer_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_5/gamma/v*
_output_shapes
: *
dtype0
¬
5Adam/transformer_block_2/layer_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_5/beta/v
ї
IAdam/transformer_block_2/layer_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_5/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp
фФ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЃФ
value£ФBЯФ BЧФ
Ѕ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
†
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
ш
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm£(m§1m•2m¶<mІ=m®>m©?m™@mЂAmђBm≠CmЃDmѓEm∞Fm±Gm≤Hm≥ImіJmµKmґLmЈMmЄ'vє(vЇ1vї2vЉ<vљ=vЊ>vњ?vј@vЅAv¬Bv√CvƒDv≈Ev∆Fv«Gv»Hv…Iv JvЋKvћLvЌMvќ
¶
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
¶
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
 
≠
Nnon_trainable_variables

	variables
trainable_variables

Olayers
regularization_losses
Player_regularization_losses
Qmetrics
Rlayer_metrics
 
b
<
embeddings
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
b
=
embeddings
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api

<0
=1

<0
=1
 
≠
[non_trainable_variables
	variables
trainable_variables

\layers
regularization_losses
]layer_regularization_losses
^metrics
_layer_metrics
ї
`_query_dense
a
_key_dense
b_value_dense
c_softmax
d_dropout_layer
e_output_dense
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
†
jlayer_with_weights-0
jlayer-0
klayer_with_weights-1
klayer-1
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
q
paxis
	Jgamma
Kbeta
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
q
uaxis
	Lgamma
Mbeta
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
R
z	variables
{trainable_variables
|regularization_losses
}	keras_api
T
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
 
≤
Вnon_trainable_variables
	variables
trainable_variables
Гlayers
regularization_losses
 Дlayer_regularization_losses
Еmetrics
Жlayer_metrics
 
 
 
≤
Зnon_trainable_variables
	variables
 trainable_variables
Иlayers
!regularization_losses
 Йlayer_regularization_losses
Кmetrics
Лlayer_metrics
 
 
 
≤
Мnon_trainable_variables
#	variables
$trainable_variables
Нlayers
%regularization_losses
 Оlayer_regularization_losses
Пmetrics
Рlayer_metrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
≤
Сnon_trainable_variables
)	variables
*trainable_variables
Тlayers
+regularization_losses
 Уlayer_regularization_losses
Фmetrics
Хlayer_metrics
 
 
 
≤
Цnon_trainable_variables
-	variables
.trainable_variables
Чlayers
/regularization_losses
 Шlayer_regularization_losses
Щmetrics
Ъlayer_metrics
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
≤
Ыnon_trainable_variables
3	variables
4trainable_variables
Ьlayers
5regularization_losses
 Эlayer_regularization_losses
Юmetrics
Яlayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_4/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_5/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_2/multi_head_attention_2/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_2/multi_head_attention_2/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_2/multi_head_attention_2/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3transformer_block_2/multi_head_attention_2/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_2/multi_head_attention_2/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_2/multi_head_attention_2/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBtransformer_block_2/multi_head_attention_2/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE@transformer_block_2/multi_head_attention_2/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_8/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_8/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_9/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_9/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_2/layer_normalization_4/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_2/layer_normalization_4/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_2/layer_normalization_5/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_2/layer_normalization_5/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7
 

†0
°1
 

<0

<0
 
≤
Ґnon_trainable_variables
S	variables
Ttrainable_variables
£layers
Uregularization_losses
 §layer_regularization_losses
•metrics
¶layer_metrics

=0

=0
 
≤
Іnon_trainable_variables
W	variables
Xtrainable_variables
®layers
Yregularization_losses
 ©layer_regularization_losses
™metrics
Ђlayer_metrics
 

0
1
 
 
 
Я
ђpartial_output_shape
≠full_output_shape

>kernel
?bias
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
Я
≤partial_output_shape
≥full_output_shape

@kernel
Abias
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
Я
Єpartial_output_shape
єfull_output_shape

Bkernel
Cbias
Ї	variables
їtrainable_variables
Љregularization_losses
љ	keras_api
V
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
V
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
Я
∆partial_output_shape
«full_output_shape

Dkernel
Ebias
»	variables
…trainable_variables
 regularization_losses
Ћ	keras_api
8
>0
?1
@2
A3
B4
C5
D6
E7
8
>0
?1
@2
A3
B4
C5
D6
E7
 
≤
ћnon_trainable_variables
f	variables
gtrainable_variables
Ќlayers
hregularization_losses
 ќlayer_regularization_losses
ѕmetrics
–layer_metrics
l

Fkernel
Gbias
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
l

Hkernel
Ibias
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api

F0
G1
H2
I3

F0
G1
H2
I3
 
≤
ўnon_trainable_variables
l	variables
mtrainable_variables
Џlayers
nregularization_losses
 џlayer_regularization_losses
№metrics
Ёlayer_metrics
 

J0
K1

J0
K1
 
≤
ёnon_trainable_variables
q	variables
rtrainable_variables
яlayers
sregularization_losses
 аlayer_regularization_losses
бmetrics
вlayer_metrics
 

L0
M1

L0
M1
 
≤
гnon_trainable_variables
v	variables
wtrainable_variables
дlayers
xregularization_losses
 еlayer_regularization_losses
жmetrics
зlayer_metrics
 
 
 
≤
иnon_trainable_variables
z	variables
{trainable_variables
йlayers
|regularization_losses
 кlayer_regularization_losses
лmetrics
мlayer_metrics
 
 
 
≥
нnon_trainable_variables
~	variables
trainable_variables
оlayers
Аregularization_losses
 пlayer_regularization_losses
рmetrics
сlayer_metrics
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

тtotal

уcount
ф	variables
х	keras_api
I

цtotal

чcount
ш
_fn_kwargs
щ	variables
ъ	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1

>0
?1
 
µ
ыnon_trainable_variables
Ѓ	variables
ѓtrainable_variables
ьlayers
∞regularization_losses
 эlayer_regularization_losses
юmetrics
€layer_metrics
 
 

@0
A1

@0
A1
 
µ
Аnon_trainable_variables
і	variables
µtrainable_variables
Бlayers
ґregularization_losses
 Вlayer_regularization_losses
Гmetrics
Дlayer_metrics
 
 

B0
C1

B0
C1
 
µ
Еnon_trainable_variables
Ї	variables
їtrainable_variables
Жlayers
Љregularization_losses
 Зlayer_regularization_losses
Иmetrics
Йlayer_metrics
 
 
 
µ
Кnon_trainable_variables
Њ	variables
њtrainable_variables
Лlayers
јregularization_losses
 Мlayer_regularization_losses
Нmetrics
Оlayer_metrics
 
 
 
µ
Пnon_trainable_variables
¬	variables
√trainable_variables
Рlayers
ƒregularization_losses
 Сlayer_regularization_losses
Тmetrics
Уlayer_metrics
 
 

D0
E1

D0
E1
 
µ
Фnon_trainable_variables
»	variables
…trainable_variables
Хlayers
 regularization_losses
 Цlayer_regularization_losses
Чmetrics
Шlayer_metrics
 
*
`0
a1
b2
c3
d4
e5
 
 
 

F0
G1

F0
G1
 
µ
Щnon_trainable_variables
—	variables
“trainable_variables
Ъlayers
”regularization_losses
 Ыlayer_regularization_losses
Ьmetrics
Эlayer_metrics

H0
I1

H0
I1
 
µ
Юnon_trainable_variables
’	variables
÷trainable_variables
Яlayers
„regularization_losses
 †layer_regularization_losses
°metrics
Ґlayer_metrics
 

j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

т0
у1

ф	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ц0
ч1

щ	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_4/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_5/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/query/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/query/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/key/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE:Adam/transformer_block_2/multi_head_attention_2/key/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/value/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/value/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
†Э
VARIABLE_VALUEGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_8/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_8/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_9/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_9/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_4/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_4/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_5/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_5/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_4/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_5/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/query/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/query/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/key/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE:Adam/transformer_block_2/multi_head_attention_2/key/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE>Adam/transformer_block_2/multi_head_attention_2/value/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE<Adam/transformer_block_2/multi_head_attention_2/value/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUEIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
†Э
VARIABLE_VALUEGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_8/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_8/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_9/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_9/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_4/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_4/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_5/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_5/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_3Placeholder*(
_output_shapes
:€€€€€€€€€и*
dtype0*
shape:€€€€€€€€€и
ё	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_35token_and_position_embedding_2/embedding_5/embeddings5token_and_position_embedding_2/embedding_4/embeddings7transformer_block_2/multi_head_attention_2/query/kernel5transformer_block_2/multi_head_attention_2/query/bias5transformer_block_2/multi_head_attention_2/key/kernel3transformer_block_2/multi_head_attention_2/key/bias7transformer_block_2/multi_head_attention_2/value/kernel5transformer_block_2/multi_head_attention_2/value/biasBtransformer_block_2/multi_head_attention_2/attention_output/kernel@transformer_block_2/multi_head_attention_2/attention_output/bias/transformer_block_2/layer_normalization_4/gamma.transformer_block_2/layer_normalization_4/betadense_8/kerneldense_8/biasdense_9/kerneldense_9/bias/transformer_block_2/layer_normalization_5/gamma.transformer_block_2/layer_normalization_5/betadense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_77408
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
„&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpKtransformer_block_2/multi_head_attention_2/query/kernel/Read/ReadVariableOpItransformer_block_2/multi_head_attention_2/query/bias/Read/ReadVariableOpItransformer_block_2/multi_head_attention_2/key/kernel/Read/ReadVariableOpGtransformer_block_2/multi_head_attention_2/key/bias/Read/ReadVariableOpKtransformer_block_2/multi_head_attention_2/value/kernel/Read/ReadVariableOpItransformer_block_2/multi_head_attention_2/value/bias/Read/ReadVariableOpVtransformer_block_2/multi_head_attention_2/attention_output/kernel/Read/ReadVariableOpTtransformer_block_2/multi_head_attention_2/attention_output/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpCtransformer_block_2/layer_normalization_4/gamma/Read/ReadVariableOpBtransformer_block_2/layer_normalization_4/beta/Read/ReadVariableOpCtransformer_block_2/layer_normalization_5/gamma/Read/ReadVariableOpBtransformer_block_2/layer_normalization_5/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/query/kernel/m/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/query/bias/m/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/key/kernel/m/Read/ReadVariableOpNAdam/transformer_block_2/multi_head_attention_2/key/bias/m/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/value/kernel/m/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/value/bias/m/Read/ReadVariableOp]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m/Read/ReadVariableOp[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_4/gamma/m/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_4/beta/m/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_5/gamma/m/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_5/beta/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/query/kernel/v/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/query/bias/v/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/key/kernel/v/Read/ReadVariableOpNAdam/transformer_block_2/multi_head_attention_2/key/bias/v/Read/ReadVariableOpRAdam/transformer_block_2/multi_head_attention_2/value/kernel/v/Read/ReadVariableOpPAdam/transformer_block_2/multi_head_attention_2/value/bias/v/Read/ReadVariableOp]Adam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v/Read/ReadVariableOp[Adam/transformer_block_2/multi_head_attention_2/attention_output/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_4/gamma/v/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_4/beta/v/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_5/gamma/v/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_5/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_78836
ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biasdense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate5token_and_position_embedding_2/embedding_4/embeddings5token_and_position_embedding_2/embedding_5/embeddings7transformer_block_2/multi_head_attention_2/query/kernel5transformer_block_2/multi_head_attention_2/query/bias5transformer_block_2/multi_head_attention_2/key/kernel3transformer_block_2/multi_head_attention_2/key/bias7transformer_block_2/multi_head_attention_2/value/kernel5transformer_block_2/multi_head_attention_2/value/biasBtransformer_block_2/multi_head_attention_2/attention_output/kernel@transformer_block_2/multi_head_attention_2/attention_output/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias/transformer_block_2/layer_normalization_4/gamma.transformer_block_2/layer_normalization_4/beta/transformer_block_2/layer_normalization_5/gamma.transformer_block_2/layer_normalization_5/betatotalcounttotal_1count_1Adam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/m<Adam/token_and_position_embedding_2/embedding_4/embeddings/m<Adam/token_and_position_embedding_2/embedding_5/embeddings/m>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m<Adam/transformer_block_2/multi_head_attention_2/query/bias/m<Adam/transformer_block_2/multi_head_attention_2/key/kernel/m:Adam/transformer_block_2/multi_head_attention_2/key/bias/m>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m<Adam/transformer_block_2/multi_head_attention_2/value/bias/mIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/mGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m6Adam/transformer_block_2/layer_normalization_4/gamma/m5Adam/transformer_block_2/layer_normalization_4/beta/m6Adam/transformer_block_2/layer_normalization_5/gamma/m5Adam/transformer_block_2/layer_normalization_5/beta/mAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v<Adam/token_and_position_embedding_2/embedding_4/embeddings/v<Adam/token_and_position_embedding_2/embedding_5/embeddings/v>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v<Adam/transformer_block_2/multi_head_attention_2/query/bias/v<Adam/transformer_block_2/multi_head_attention_2/key/kernel/v:Adam/transformer_block_2/multi_head_attention_2/key/bias/v>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v<Adam/transformer_block_2/multi_head_attention_2/value/bias/vIAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/vGAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v6Adam/transformer_block_2/layer_normalization_4/gamma/v5Adam/transformer_block_2/layer_normalization_4/beta/v6Adam/transformer_block_2/layer_normalization_5/gamma/v5Adam/transformer_block_2/layer_normalization_5/beta/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_79071Рэ
і№
—
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_78179

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityИҐ.layer_normalization_4/batchnorm/ReadVariableOpҐ2layer_normalization_4/batchnorm/mul/ReadVariableOpҐ.layer_normalization_5/batchnorm/ReadVariableOpҐ2layer_normalization_5/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_2/attention_output/add/ReadVariableOpҐDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_2/key/add/ReadVariableOpҐ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/query/add/ReadVariableOpҐ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/value/add/ReadVariableOpҐ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐ+sequential_2/dense_8/BiasAdd/ReadVariableOpҐ-sequential_2/dense_8/Tensordot/ReadVariableOpҐ+sequential_2/dense_9/BiasAdd/ReadVariableOpҐ-sequential_2/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumџ
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpц
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/query/addч
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsum’
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpо
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2 
multi_head_attention_2/key/addэ
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumџ
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpц
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/value/addБ
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_2/Mul/y«
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2
multi_head_attention_2/Mulю
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/Einsum∆
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2(
&multi_head_attention_2/softmax/Softmaxћ
'multi_head_attention_2/dropout/IdentityIdentity0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2)
'multi_head_attention_2/dropout/IdentityХ
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/Identity:output:0$multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/EinsumЮ
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumш
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOpЮ
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2-
+multi_head_attention_2/attention_output/addЬ
dropout_8/IdentityIdentity/multi_head_attention_2/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/Identityo
addAddV2inputsdropout_8/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
addґ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesа
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_4/moments/meanћ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_4/moments/StopGradientм
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_4/moments/SquaredDifferenceЊ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesШ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_4/moments/varianceУ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_4/batchnorm/add/yл
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_4/batchnorm/addЈ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_4/batchnorm/Rsqrtа
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpп
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/mulЊ
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_1в
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_2‘
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpл
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/subв
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/add_1’
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOpФ
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axesЫ
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/free•
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/ShapeЮ
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisЇ
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ґ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisј
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1Ц
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Const‘
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/ProdЪ
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1№
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1Ъ
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axisЩ
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatа
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackу
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2*
(sequential_2/dense_8/Tensordot/transposeу
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_8/Tensordot/Reshapeт
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2'
%sequential_2/dense_8/Tensordot/MatMulЪ
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2(
&sequential_2/dense_8/Tensordot/Const_2Ю
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axis¶
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1е
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2 
sequential_2/dense_8/TensordotЋ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOp№
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/BiasAddЬ
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/Relu’
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOpФ
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axesЫ
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/free£
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/ShapeЮ
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisЇ
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ґ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisј
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1Ц
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Const‘
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/ProdЪ
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1№
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1Ъ
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axisЩ
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatа
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stackс
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2*
(sequential_2/dense_9/Tensordot/transposeу
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_9/Tensordot/Reshapeт
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/dense_9/Tensordot/MatMulЪ
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2Ю
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axis¶
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1е
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
sequential_2/dense_9/TensordotЋ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOp№
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
sequential_2/dense_9/BiasAddТ
dropout_9/IdentityIdentity%sequential_2/dense_9/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/IdentityЦ
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
add_1ґ
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesв
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_5/moments/meanћ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_5/moments/StopGradientо
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_5/moments/SquaredDifferenceЊ
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indicesШ
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_5/moments/varianceУ
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_5/batchnorm/add/yл
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_5/batchnorm/addЈ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_5/batchnorm/Rsqrtа
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpп
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/mulј
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_1в
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_2‘
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpл
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/subв
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/add_1‘
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€и ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
ЦЯ
Л
B__inference_model_2_layer_call_and_return_conditional_losses_77608

inputsE
Atoken_and_position_embedding_2_embedding_5_embedding_lookup_77419E
Atoken_and_position_embedding_2_embedding_4_embedding_lookup_77425Z
Vtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_query_add_readvariableop_resourceX
Ttransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resourceZ
Vtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_value_add_readvariableop_resourcee
atransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resourceS
Otransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resourceS
Otransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityИҐdense_10/BiasAdd/ReadVariableOpҐdense_10/MatMul/ReadVariableOpҐdense_11/BiasAdd/ReadVariableOpҐdense_11/MatMul/ReadVariableOpҐ;token_and_position_embedding_2/embedding_4/embedding_lookupҐ;token_and_position_embedding_2/embedding_5/embedding_lookupҐBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpҐFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpҐBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpҐFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpҐNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpҐXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpҐKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpҐMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpҐMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐ?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpҐAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpҐ?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpҐAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpВ
$token_and_position_embedding_2/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shapeї
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€24
2token_and_position_embedding_2/strided_slice/stackґ
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1ґ
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2Ь
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_sliceЪ
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/startЪ
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/deltaЫ
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2&
$token_and_position_embedding_2/range»
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_5_embedding_lookup_77419-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/77419*'
_output_shapes
:€€€€€€€€€ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookupФ
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/77419*'
_output_shapes
:€€€€€€€€€ 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityЭ
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1і
/token_and_position_embedding_2/embedding_4/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€и21
/token_and_position_embedding_2/embedding_4/Cast”
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_4_embedding_lookup_774253token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/77425*,
_output_shapes
:€€€€€€€€€и *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookupЩ
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/77425*,
_output_shapes
:€€€€€€€€€и 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityҐ
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1™
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2$
"token_and_position_embedding_2/addє
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02O
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpк
>transformer_block_2/multi_head_attention_2/query/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/query/einsum/EinsumЧ
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype02E
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp∆
4transformer_block_2/multi_head_attention_2/query/addAddV2Gtransformer_block_2/multi_head_attention_2/query/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 26
4transformer_block_2/multi_head_attention_2/query/add≥
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02M
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpд
<transformer_block_2/multi_head_attention_2/key/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Stransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2>
<transformer_block_2/multi_head_attention_2/key/einsum/EinsumС
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpReadVariableOpJtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02C
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpЊ
2transformer_block_2/multi_head_attention_2/key/addAddV2Etransformer_block_2/multi_head_attention_2/key/einsum/Einsum:output:0Itransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 24
2transformer_block_2/multi_head_attention_2/key/addє
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02O
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpк
>transformer_block_2/multi_head_attention_2/value/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/value/einsum/EinsumЧ
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype02E
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp∆
4transformer_block_2/multi_head_attention_2/value/addAddV2Gtransformer_block_2/multi_head_attention_2/value/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 26
4transformer_block_2/multi_head_attention_2/value/add©
0transformer_block_2/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>22
0transformer_block_2/multi_head_attention_2/Mul/yЧ
.transformer_block_2/multi_head_attention_2/MulMul8transformer_block_2/multi_head_attention_2/query/add:z:09transformer_block_2/multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 20
.transformer_block_2/multi_head_attention_2/Mulќ
8transformer_block_2/multi_head_attention_2/einsum/EinsumEinsum6transformer_block_2/multi_head_attention_2/key/add:z:02transformer_block_2/multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2:
8transformer_block_2/multi_head_attention_2/einsum/EinsumВ
:transformer_block_2/multi_head_attention_2/softmax/SoftmaxSoftmaxAtransformer_block_2/multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2<
:transformer_block_2/multi_head_attention_2/softmax/Softmax…
@transformer_block_2/multi_head_attention_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2B
@transformer_block_2/multi_head_attention_2/dropout/dropout/Const‘
>transformer_block_2/multi_head_attention_2/dropout/dropout/MulMulDtransformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0Itransformer_block_2/multi_head_attention_2/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2@
>transformer_block_2/multi_head_attention_2/dropout/dropout/Mulш
@transformer_block_2/multi_head_attention_2/dropout/dropout/ShapeShapeDtransformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_2/multi_head_attention_2/dropout/dropout/Shape„
Wtransformer_block_2/multi_head_attention_2/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_2/multi_head_attention_2/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии*
dtype02Y
Wtransformer_block_2/multi_head_attention_2/dropout/dropout/random_uniform/RandomUniformџ
Itransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual/yФ
Gtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_2/multi_head_attention_2/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2I
Gtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqualҐ
?transformer_block_2/multi_head_attention_2/dropout/dropout/CastCastKtransformer_block_2/multi_head_attention_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€
ии2A
?transformer_block_2/multi_head_attention_2/dropout/dropout/Cast–
@transformer_block_2/multi_head_attention_2/dropout/dropout/Mul_1MulBtransformer_block_2/multi_head_attention_2/dropout/dropout/Mul:z:0Ctransformer_block_2/multi_head_attention_2/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2B
@transformer_block_2/multi_head_attention_2/dropout/dropout/Mul_1е
:transformer_block_2/multi_head_attention_2/einsum_1/EinsumEinsumDtransformer_block_2/multi_head_attention_2/dropout/dropout/Mul_1:z:08transformer_block_2/multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2<
:transformer_block_2/multi_head_attention_2/einsum_1/EinsumЏ
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02Z
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp§
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumEinsumCtransformer_block_2/multi_head_attention_2/einsum_1/Einsum:output:0`transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe2K
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsumі
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpо
?transformer_block_2/multi_head_attention_2/attention_output/addAddV2Rtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum:output:0Vtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2A
?transformer_block_2/multi_head_attention_2/attention_output/addЯ
+transformer_block_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2-
+transformer_block_2/dropout_8/dropout/ConstП
)transformer_block_2/dropout_8/dropout/MulMulCtransformer_block_2/multi_head_attention_2/attention_output/add:z:04transformer_block_2/dropout_8/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2+
)transformer_block_2/dropout_8/dropout/MulЌ
+transformer_block_2/dropout_8/dropout/ShapeShapeCtransformer_block_2/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_2/dropout_8/dropout/ShapeУ
Btransformer_block_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_2/dropout_8/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€и *
dtype02D
Btransformer_block_2/dropout_8/dropout/random_uniform/RandomUniform±
4transformer_block_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=26
4transformer_block_2/dropout_8/dropout/GreaterEqual/yї
2transformer_block_2/dropout_8/dropout/GreaterEqualGreaterEqualKtransformer_block_2/dropout_8/dropout/random_uniform/RandomUniform:output:0=transformer_block_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 24
2transformer_block_2/dropout_8/dropout/GreaterEqualё
*transformer_block_2/dropout_8/dropout/CastCast6transformer_block_2/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€и 2,
*transformer_block_2/dropout_8/dropout/Castч
+transformer_block_2/dropout_8/dropout/Mul_1Mul-transformer_block_2/dropout_8/dropout/Mul:z:0.transformer_block_2/dropout_8/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€и 2-
+transformer_block_2/dropout_8/dropout/Mul_1Ћ
transformer_block_2/addAddV2&token_and_position_embedding_2/add:z:0/transformer_block_2/dropout_8/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
transformer_block_2/addё
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indices∞
6transformer_block_2/layer_normalization_4/moments/meanMeantransformer_block_2/add:z:0Qtransformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(28
6transformer_block_2/layer_normalization_4/moments/meanИ
>transformer_block_2/layer_normalization_4/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2@
>transformer_block_2/layer_normalization_4/moments/StopGradientЉ
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add:z:0Gtransformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2E
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceж
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesи
:transformer_block_2/layer_normalization_4/moments/varianceMeanGtransformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2<
:transformer_block_2/layer_normalization_4/moments/varianceї
9transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_2/layer_normalization_4/batchnorm/add/yї
7transformer_block_2/layer_normalization_4/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_4/moments/variance:output:0Btransformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и29
7transformer_block_2/layer_normalization_4/batchnorm/addу
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2;
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtЬ
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpњ
7transformer_block_2/layer_normalization_4/batchnorm/mulMul=transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_4/batchnorm/mulО
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Multransformer_block_2/add:z:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_1≤
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_4/moments/mean:output:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Р
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpї
7transformer_block_2/layer_normalization_4/batchnorm/subSubJtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_4/batchnorm/sub≤
9transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_4/batchnorm/add_1С
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02C
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpЉ
7transformer_block_2/sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_8/Tensordot/axes√
7transformer_block_2/sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_8/Tensordot/freeб
8transformer_block_2/sequential_2/dense_8/Tensordot/ShapeShape=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/Shape∆
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisЮ
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2 
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis§
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1Њ
8transformer_block_2/sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_8/Tensordot/Const§
7transformer_block_2/sequential_2/dense_8/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_8/Tensordot/Prod¬
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1ђ
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1¬
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisэ
9transformer_block_2/sequential_2/dense_8/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_8/Tensordot/concat∞
8transformer_block_2/sequential_2/dense_8/Tensordot/stackPack@transformer_block_2/sequential_2/dense_8/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/stack√
<transformer_block_2/sequential_2/dense_8/Tensordot/transpose	Transpose=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Btransformer_block_2/sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2>
<transformer_block_2/sequential_2/dense_8/Tensordot/transpose√
:transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_8/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Reshape¬
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_8/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2;
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMul¬
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2∆
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisК
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1µ
2transformer_block_2/sequential_2/dense_8/TensordotReshapeCtransformer_block_2/sequential_2/dense_8/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd24
2transformer_block_2/sequential_2/dense_8/TensordotЗ
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02A
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpђ
0transformer_block_2/sequential_2/dense_8/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_8/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd22
0transformer_block_2/sequential_2/dense_8/BiasAddЎ
-transformer_block_2/sequential_2/dense_8/ReluRelu9transformer_block_2/sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2/
-transformer_block_2/sequential_2/dense_8/ReluС
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02C
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpЉ
7transformer_block_2/sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_9/Tensordot/axes√
7transformer_block_2/sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_9/Tensordot/freeя
8transformer_block_2/sequential_2/dense_9/Tensordot/ShapeShape;transformer_block_2/sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/Shape∆
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisЮ
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2 
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis§
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1Њ
8transformer_block_2/sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_9/Tensordot/Const§
7transformer_block_2/sequential_2/dense_9/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_9/Tensordot/Prod¬
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1ђ
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1¬
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisэ
9transformer_block_2/sequential_2/dense_9/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_9/Tensordot/concat∞
8transformer_block_2/sequential_2/dense_9/Tensordot/stackPack@transformer_block_2/sequential_2/dense_9/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/stackЅ
<transformer_block_2/sequential_2/dense_9/Tensordot/transpose	Transpose;transformer_block_2/sequential_2/dense_8/Relu:activations:0Btransformer_block_2/sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2>
<transformer_block_2/sequential_2/dense_9/Tensordot/transpose√
:transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_9/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Reshape¬
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_9/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMul¬
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2∆
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisК
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1µ
2transformer_block_2/sequential_2/dense_9/TensordotReshapeCtransformer_block_2/sequential_2/dense_9/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 24
2transformer_block_2/sequential_2/dense_9/TensordotЗ
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpђ
0transformer_block_2/sequential_2/dense_9/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_9/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 22
0transformer_block_2/sequential_2/dense_9/BiasAddЯ
+transformer_block_2/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2-
+transformer_block_2/dropout_9/dropout/ConstЕ
)transformer_block_2/dropout_9/dropout/MulMul9transformer_block_2/sequential_2/dense_9/BiasAdd:output:04transformer_block_2/dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2+
)transformer_block_2/dropout_9/dropout/Mul√
+transformer_block_2/dropout_9/dropout/ShapeShape9transformer_block_2/sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_2/dropout_9/dropout/ShapeУ
Btransformer_block_2/dropout_9/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_2/dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€и *
dtype02D
Btransformer_block_2/dropout_9/dropout/random_uniform/RandomUniform±
4transformer_block_2/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=26
4transformer_block_2/dropout_9/dropout/GreaterEqual/yї
2transformer_block_2/dropout_9/dropout/GreaterEqualGreaterEqualKtransformer_block_2/dropout_9/dropout/random_uniform/RandomUniform:output:0=transformer_block_2/dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 24
2transformer_block_2/dropout_9/dropout/GreaterEqualё
*transformer_block_2/dropout_9/dropout/CastCast6transformer_block_2/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€и 2,
*transformer_block_2/dropout_9/dropout/Castч
+transformer_block_2/dropout_9/dropout/Mul_1Mul-transformer_block_2/dropout_9/dropout/Mul:z:0.transformer_block_2/dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€и 2-
+transformer_block_2/dropout_9/dropout/Mul_1ж
transformer_block_2/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0/transformer_block_2/dropout_9/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
transformer_block_2/add_1ё
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indices≤
6transformer_block_2/layer_normalization_5/moments/meanMeantransformer_block_2/add_1:z:0Qtransformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(28
6transformer_block_2/layer_normalization_5/moments/meanИ
>transformer_block_2/layer_normalization_5/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2@
>transformer_block_2/layer_normalization_5/moments/StopGradientЊ
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add_1:z:0Gtransformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2E
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceж
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesи
:transformer_block_2/layer_normalization_5/moments/varianceMeanGtransformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2<
:transformer_block_2/layer_normalization_5/moments/varianceї
9transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_2/layer_normalization_5/batchnorm/add/yї
7transformer_block_2/layer_normalization_5/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_5/moments/variance:output:0Btransformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и29
7transformer_block_2/layer_normalization_5/batchnorm/addу
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2;
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtЬ
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpњ
7transformer_block_2/layer_normalization_5/batchnorm/mulMul=transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_5/batchnorm/mulР
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Multransformer_block_2/add_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_1≤
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_5/moments/mean:output:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Р
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpї
7transformer_block_2/layer_normalization_5/batchnorm/subSubJtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_5/batchnorm/sub≤
9transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_5/batchnorm/add_1®
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesч
global_average_pooling1d_2/MeanMean=transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
global_average_pooling1d_2/Meany
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout_10/dropout/Constґ
dropout_10/dropout/MulMul(global_average_pooling1d_2/Mean:output:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_10/dropout/MulМ
dropout_10/dropout/ShapeShape(global_average_pooling1d_2/Mean:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape’
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype021
/dropout_10/dropout/random_uniform/RandomUniformЛ
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2#
!dropout_10/dropout/GreaterEqual/yк
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
dropout_10/dropout/GreaterEqual†
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_10/dropout/Cast¶
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_10/dropout/Mul_1®
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_10/MatMul/ReadVariableOp§
dense_10/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_10/MatMulІ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp•
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_10/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout_11/dropout/Const©
dropout_11/dropout/MulMuldense_10/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape’
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype021
/dropout_11/dropout/random_uniform/RandomUniformЛ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2#
!dropout_11/dropout/GreaterEqual/yк
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
dropout_11/dropout/GreaterEqual†
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
dropout_11/dropout/Cast¶
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout_11/dropout/Mul_1®
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp§
dense_11/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/MatMulІ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp•
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/SoftmaxВ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupC^transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpC^transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpO^transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpY^transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpL^transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp@^transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp@^transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2И
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2Р
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2И
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2Р
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp2і
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp2Ъ
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp2Ю
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp2Ю
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2В
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp2Ж
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp2В
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp2Ж
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
п
|
'__inference_dense_8_layer_call_fn_78549

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€иd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_763792
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€иd2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€и ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
Б
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_78287

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
¶
,__inference_sequential_2_layer_call_fn_76511
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_765002
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:€€€€€€€€€и 
'
_user_specified_namedense_8_input
»
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_78339

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_78292

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Б
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_78334

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Б
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_76982

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ё
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_76963

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€и :T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
“

я
3__inference_transformer_block_2_layer_call_fn_78216

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_767222
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€и ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
ть
—
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_76722

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityИҐ.layer_normalization_4/batchnorm/ReadVariableOpҐ2layer_normalization_4/batchnorm/mul/ReadVariableOpҐ.layer_normalization_5/batchnorm/ReadVariableOpҐ2layer_normalization_5/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_2/attention_output/add/ReadVariableOpҐDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_2/key/add/ReadVariableOpҐ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/query/add/ReadVariableOpҐ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/value/add/ReadVariableOpҐ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐ+sequential_2/dense_8/BiasAdd/ReadVariableOpҐ-sequential_2/dense_8/Tensordot/ReadVariableOpҐ+sequential_2/dense_9/BiasAdd/ReadVariableOpҐ-sequential_2/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumџ
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpц
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/query/addч
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsum’
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpо
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2 
multi_head_attention_2/key/addэ
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumџ
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpц
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/value/addБ
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_2/Mul/y«
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2
multi_head_attention_2/Mulю
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/Einsum∆
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2(
&multi_head_attention_2/softmax/Softmax°
,multi_head_attention_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_2/dropout/dropout/ConstД
*multi_head_attention_2/dropout/dropout/MulMul0multi_head_attention_2/softmax/Softmax:softmax:05multi_head_attention_2/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2,
*multi_head_attention_2/dropout/dropout/MulЉ
,multi_head_attention_2/dropout/dropout/ShapeShape0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_2/dropout/dropout/ShapeЫ
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_2/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии*
dtype02E
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_2/dropout/dropout/GreaterEqual/yƒ
3multi_head_attention_2/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_2/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии25
3multi_head_attention_2/dropout/dropout/GreaterEqualж
+multi_head_attention_2/dropout/dropout/CastCast7multi_head_attention_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€
ии2-
+multi_head_attention_2/dropout/dropout/CastА
,multi_head_attention_2/dropout/dropout/Mul_1Mul.multi_head_attention_2/dropout/dropout/Mul:z:0/multi_head_attention_2/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2.
,multi_head_attention_2/dropout/dropout/Mul_1Х
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/dropout/Mul_1:z:0$multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/EinsumЮ
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumш
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOpЮ
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2-
+multi_head_attention_2/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_8/dropout/Constњ
dropout_8/dropout/MulMul/multi_head_attention_2/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/dropout/MulС
dropout_8/dropout/ShapeShape/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape„
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€и *
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЙ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_8/dropout/GreaterEqual/yл
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
dropout_8/dropout/GreaterEqualҐ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/dropout/CastІ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/dropout/Mul_1o
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
addґ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesа
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_4/moments/meanћ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_4/moments/StopGradientм
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_4/moments/SquaredDifferenceЊ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesШ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_4/moments/varianceУ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_4/batchnorm/add/yл
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_4/batchnorm/addЈ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_4/batchnorm/Rsqrtа
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpп
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/mulЊ
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_1в
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_2‘
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpл
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/subв
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/add_1’
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOpФ
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axesЫ
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/free•
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/ShapeЮ
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisЇ
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ґ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisј
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1Ц
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Const‘
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/ProdЪ
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1№
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1Ъ
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axisЩ
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatа
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackу
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2*
(sequential_2/dense_8/Tensordot/transposeу
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_8/Tensordot/Reshapeт
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2'
%sequential_2/dense_8/Tensordot/MatMulЪ
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2(
&sequential_2/dense_8/Tensordot/Const_2Ю
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axis¶
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1е
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2 
sequential_2/dense_8/TensordotЋ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOp№
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/BiasAddЬ
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/Relu’
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOpФ
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axesЫ
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/free£
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/ShapeЮ
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisЇ
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ґ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisј
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1Ц
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Const‘
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/ProdЪ
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1№
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1Ъ
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axisЩ
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatа
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stackс
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2*
(sequential_2/dense_9/Tensordot/transposeу
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_9/Tensordot/Reshapeт
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/dense_9/Tensordot/MatMulЪ
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2Ю
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axis¶
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1е
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
sequential_2/dense_9/TensordotЋ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOp№
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
sequential_2/dense_9/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_9/dropout/Constµ
dropout_9/dropout/MulMul%sequential_2/dense_9/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/dropout/MulЗ
dropout_9/dropout/ShapeShape%sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape„
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€и *
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЙ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_9/dropout/GreaterEqual/yл
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
dropout_9/dropout/GreaterEqualҐ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/dropout/CastІ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/dropout/Mul_1Ц
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
add_1ґ
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesв
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_5/moments/meanћ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_5/moments/StopGradientо
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_5/moments/SquaredDifferenceЊ
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indicesШ
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_5/moments/varianceУ
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_5/batchnorm/add/yл
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_5/batchnorm/addЈ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_5/batchnorm/Rsqrtа
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpп
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/mulј
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_1в
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_2‘
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpл
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/subв
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/add_1‘
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€и ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
ћћ
љ3
!__inference__traced_restore_79071
file_prefix$
 assignvariableop_dense_10_kernel$
 assignvariableop_1_dense_10_bias&
"assignvariableop_2_dense_11_kernel$
 assignvariableop_3_dense_11_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rateL
Hassignvariableop_9_token_and_position_embedding_2_embedding_4_embeddingsM
Iassignvariableop_10_token_and_position_embedding_2_embedding_5_embeddingsO
Kassignvariableop_11_transformer_block_2_multi_head_attention_2_query_kernelM
Iassignvariableop_12_transformer_block_2_multi_head_attention_2_query_biasM
Iassignvariableop_13_transformer_block_2_multi_head_attention_2_key_kernelK
Gassignvariableop_14_transformer_block_2_multi_head_attention_2_key_biasO
Kassignvariableop_15_transformer_block_2_multi_head_attention_2_value_kernelM
Iassignvariableop_16_transformer_block_2_multi_head_attention_2_value_biasZ
Vassignvariableop_17_transformer_block_2_multi_head_attention_2_attention_output_kernelX
Tassignvariableop_18_transformer_block_2_multi_head_attention_2_attention_output_bias&
"assignvariableop_19_dense_8_kernel$
 assignvariableop_20_dense_8_bias&
"assignvariableop_21_dense_9_kernel$
 assignvariableop_22_dense_9_biasG
Cassignvariableop_23_transformer_block_2_layer_normalization_4_gammaF
Bassignvariableop_24_transformer_block_2_layer_normalization_4_betaG
Cassignvariableop_25_transformer_block_2_layer_normalization_5_gammaF
Bassignvariableop_26_transformer_block_2_layer_normalization_5_beta
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1.
*assignvariableop_31_adam_dense_10_kernel_m,
(assignvariableop_32_adam_dense_10_bias_m.
*assignvariableop_33_adam_dense_11_kernel_m,
(assignvariableop_34_adam_dense_11_bias_mT
Passignvariableop_35_adam_token_and_position_embedding_2_embedding_4_embeddings_mT
Passignvariableop_36_adam_token_and_position_embedding_2_embedding_5_embeddings_mV
Rassignvariableop_37_adam_transformer_block_2_multi_head_attention_2_query_kernel_mT
Passignvariableop_38_adam_transformer_block_2_multi_head_attention_2_query_bias_mT
Passignvariableop_39_adam_transformer_block_2_multi_head_attention_2_key_kernel_mR
Nassignvariableop_40_adam_transformer_block_2_multi_head_attention_2_key_bias_mV
Rassignvariableop_41_adam_transformer_block_2_multi_head_attention_2_value_kernel_mT
Passignvariableop_42_adam_transformer_block_2_multi_head_attention_2_value_bias_ma
]assignvariableop_43_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_m_
[assignvariableop_44_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_m-
)assignvariableop_45_adam_dense_8_kernel_m+
'assignvariableop_46_adam_dense_8_bias_m-
)assignvariableop_47_adam_dense_9_kernel_m+
'assignvariableop_48_adam_dense_9_bias_mN
Jassignvariableop_49_adam_transformer_block_2_layer_normalization_4_gamma_mM
Iassignvariableop_50_adam_transformer_block_2_layer_normalization_4_beta_mN
Jassignvariableop_51_adam_transformer_block_2_layer_normalization_5_gamma_mM
Iassignvariableop_52_adam_transformer_block_2_layer_normalization_5_beta_m.
*assignvariableop_53_adam_dense_10_kernel_v,
(assignvariableop_54_adam_dense_10_bias_v.
*assignvariableop_55_adam_dense_11_kernel_v,
(assignvariableop_56_adam_dense_11_bias_vT
Passignvariableop_57_adam_token_and_position_embedding_2_embedding_4_embeddings_vT
Passignvariableop_58_adam_token_and_position_embedding_2_embedding_5_embeddings_vV
Rassignvariableop_59_adam_transformer_block_2_multi_head_attention_2_query_kernel_vT
Passignvariableop_60_adam_transformer_block_2_multi_head_attention_2_query_bias_vT
Passignvariableop_61_adam_transformer_block_2_multi_head_attention_2_key_kernel_vR
Nassignvariableop_62_adam_transformer_block_2_multi_head_attention_2_key_bias_vV
Rassignvariableop_63_adam_transformer_block_2_multi_head_attention_2_value_kernel_vT
Passignvariableop_64_adam_transformer_block_2_multi_head_attention_2_value_bias_va
]assignvariableop_65_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_v_
[assignvariableop_66_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_v-
)assignvariableop_67_adam_dense_8_kernel_v+
'assignvariableop_68_adam_dense_8_bias_v-
)assignvariableop_69_adam_dense_9_kernel_v+
'assignvariableop_70_adam_dense_9_bias_vN
Jassignvariableop_71_adam_transformer_block_2_layer_normalization_4_gamma_vM
Iassignvariableop_72_adam_transformer_block_2_layer_normalization_4_beta_vN
Jassignvariableop_73_adam_transformer_block_2_layer_normalization_5_gamma_vM
Iassignvariableop_74_adam_transformer_block_2_layer_normalization_5_beta_v
identity_76ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_8ҐAssignVariableOp_9–$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*№#
value“#Bѕ#LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names©
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*≠
value£B†LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices™
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*∆
_output_shapes≥
∞::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ґ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8™
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ќ
AssignVariableOp_9AssignVariableOpHassignvariableop_9_token_and_position_embedding_2_embedding_4_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10—
AssignVariableOp_10AssignVariableOpIassignvariableop_10_token_and_position_embedding_2_embedding_5_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11”
AssignVariableOp_11AssignVariableOpKassignvariableop_11_transformer_block_2_multi_head_attention_2_query_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12—
AssignVariableOp_12AssignVariableOpIassignvariableop_12_transformer_block_2_multi_head_attention_2_query_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13—
AssignVariableOp_13AssignVariableOpIassignvariableop_13_transformer_block_2_multi_head_attention_2_key_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ѕ
AssignVariableOp_14AssignVariableOpGassignvariableop_14_transformer_block_2_multi_head_attention_2_key_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15”
AssignVariableOp_15AssignVariableOpKassignvariableop_15_transformer_block_2_multi_head_attention_2_value_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16—
AssignVariableOp_16AssignVariableOpIassignvariableop_16_transformer_block_2_multi_head_attention_2_value_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ё
AssignVariableOp_17AssignVariableOpVassignvariableop_17_transformer_block_2_multi_head_attention_2_attention_output_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18№
AssignVariableOp_18AssignVariableOpTassignvariableop_18_transformer_block_2_multi_head_attention_2_attention_output_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19™
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_8_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_8_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21™
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_9_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22®
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_9_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ћ
AssignVariableOp_23AssignVariableOpCassignvariableop_23_transformer_block_2_layer_normalization_4_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24 
AssignVariableOp_24AssignVariableOpBassignvariableop_24_transformer_block_2_layer_normalization_4_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ћ
AssignVariableOp_25AssignVariableOpCassignvariableop_25_transformer_block_2_layer_normalization_5_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26 
AssignVariableOp_26AssignVariableOpBassignvariableop_26_transformer_block_2_layer_normalization_5_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27°
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30£
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≤
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_10_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32∞
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_10_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33≤
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_11_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34∞
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_11_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ў
AssignVariableOp_35AssignVariableOpPassignvariableop_35_adam_token_and_position_embedding_2_embedding_4_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ў
AssignVariableOp_36AssignVariableOpPassignvariableop_36_adam_token_and_position_embedding_2_embedding_5_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Џ
AssignVariableOp_37AssignVariableOpRassignvariableop_37_adam_transformer_block_2_multi_head_attention_2_query_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ў
AssignVariableOp_38AssignVariableOpPassignvariableop_38_adam_transformer_block_2_multi_head_attention_2_query_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ў
AssignVariableOp_39AssignVariableOpPassignvariableop_39_adam_transformer_block_2_multi_head_attention_2_key_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40÷
AssignVariableOp_40AssignVariableOpNassignvariableop_40_adam_transformer_block_2_multi_head_attention_2_key_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Џ
AssignVariableOp_41AssignVariableOpRassignvariableop_41_adam_transformer_block_2_multi_head_attention_2_value_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ў
AssignVariableOp_42AssignVariableOpPassignvariableop_42_adam_transformer_block_2_multi_head_attention_2_value_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43е
AssignVariableOp_43AssignVariableOp]assignvariableop_43_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44г
AssignVariableOp_44AssignVariableOp[assignvariableop_44_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45±
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_8_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46ѓ
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_8_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47±
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_9_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ѓ
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_9_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49“
AssignVariableOp_49AssignVariableOpJassignvariableop_49_adam_transformer_block_2_layer_normalization_4_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50—
AssignVariableOp_50AssignVariableOpIassignvariableop_50_adam_transformer_block_2_layer_normalization_4_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51“
AssignVariableOp_51AssignVariableOpJassignvariableop_51_adam_transformer_block_2_layer_normalization_5_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52—
AssignVariableOp_52AssignVariableOpIassignvariableop_52_adam_transformer_block_2_layer_normalization_5_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53≤
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_10_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54∞
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_10_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55≤
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_11_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56∞
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_11_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ў
AssignVariableOp_57AssignVariableOpPassignvariableop_57_adam_token_and_position_embedding_2_embedding_4_embeddings_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ў
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adam_token_and_position_embedding_2_embedding_5_embeddings_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Џ
AssignVariableOp_59AssignVariableOpRassignvariableop_59_adam_transformer_block_2_multi_head_attention_2_query_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ў
AssignVariableOp_60AssignVariableOpPassignvariableop_60_adam_transformer_block_2_multi_head_attention_2_query_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ў
AssignVariableOp_61AssignVariableOpPassignvariableop_61_adam_transformer_block_2_multi_head_attention_2_key_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62÷
AssignVariableOp_62AssignVariableOpNassignvariableop_62_adam_transformer_block_2_multi_head_attention_2_key_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Џ
AssignVariableOp_63AssignVariableOpRassignvariableop_63_adam_transformer_block_2_multi_head_attention_2_value_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ў
AssignVariableOp_64AssignVariableOpPassignvariableop_64_adam_transformer_block_2_multi_head_attention_2_value_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65е
AssignVariableOp_65AssignVariableOp]assignvariableop_65_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66г
AssignVariableOp_66AssignVariableOp[assignvariableop_66_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67±
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_dense_8_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68ѓ
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_dense_8_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69±
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_9_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70ѓ
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_9_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71“
AssignVariableOp_71AssignVariableOpJassignvariableop_71_adam_transformer_block_2_layer_normalization_4_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72—
AssignVariableOp_72AssignVariableOpIassignvariableop_72_adam_transformer_block_2_layer_normalization_4_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73“
AssignVariableOp_73AssignVariableOpJassignvariableop_73_adam_transformer_block_2_layer_normalization_5_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74—
AssignVariableOp_74AssignVariableOpIassignvariableop_74_adam_transformer_block_2_layer_normalization_5_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp–
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75√
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*√
_input_shapes±
Ѓ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
л
Б
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_77895
x&
"embedding_5_embedding_lookup_77882&
"embedding_4_embedding_lookup_77888
identityИҐembedding_4/embedding_lookupҐembedding_5/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
range≠
embedding_5/embedding_lookupResourceGather"embedding_5_embedding_lookup_77882range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_5/embedding_lookup/77882*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_5/embedding_lookupШ
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_5/embedding_lookup/77882*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_5/embedding_lookup/Identityј
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_5/embedding_lookup/Identity_1q
embedding_4/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€и2
embedding_4/CastЄ
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_77888embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/77888*,
_output_shapes
:€€€€€€€€€и *
dtype02
embedding_4/embedding_lookupЭ
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/77888*,
_output_shapes
:€€€€€€€€€и 2'
%embedding_4/embedding_lookup/Identity≈
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2)
'embedding_4/embedding_lookup/Identity_1Ѓ
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
addЮ
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€и

_user_specified_namex
А*
£
B__inference_model_2_layer_call_and_return_conditional_losses_77140
input_3(
$token_and_position_embedding_2_77088(
$token_and_position_embedding_2_77090
transformer_block_2_77093
transformer_block_2_77095
transformer_block_2_77097
transformer_block_2_77099
transformer_block_2_77101
transformer_block_2_77103
transformer_block_2_77105
transformer_block_2_77107
transformer_block_2_77109
transformer_block_2_77111
transformer_block_2_77113
transformer_block_2_77115
transformer_block_2_77117
transformer_block_2_77119
transformer_block_2_77121
transformer_block_2_77123
dense_10_77128
dense_10_77130
dense_11_77134
dense_11_77136
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ6token_and_position_embedding_2/StatefulPartitionedCallҐ+transformer_block_2/StatefulPartitionedCallИ
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3$token_and_position_embedding_2_77088$token_and_position_embedding_2_77090*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_7655828
6token_and_position_embedding_2/StatefulPartitionedCallЯ
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_77093transformer_block_2_77095transformer_block_2_77097transformer_block_2_77099transformer_block_2_77101transformer_block_2_77103transformer_block_2_77105transformer_block_2_77107transformer_block_2_77109transformer_block_2_77111transformer_block_2_77113transformer_block_2_77115transformer_block_2_77117transformer_block_2_77119transformer_block_2_77121transformer_block_2_77123*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_768492-
+transformer_block_2/StatefulPartitionedCallЇ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_769632,
*global_average_pooling1d_2/PartitionedCallЙ
dropout_10/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_769872
dropout_10/PartitionedCall±
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_10_77128dense_10_77130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_770112"
 dense_10/StatefulPartitionedCall€
dropout_11/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_770442
dropout_11/PartitionedCall±
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_11_77134dense_11_77136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_770682"
 dense_11/StatefulPartitionedCall™
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€и
!
_user_specified_name	input_3
ѕ
¶
,__inference_sequential_2_layer_call_fn_76484
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_764732
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:€€€€€€€€€и 
'
_user_specified_namedense_8_input
Ї
Я
,__inference_sequential_2_layer_call_fn_78496

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_764732
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
х	
№
C__inference_dense_11_layer_call_and_return_conditional_losses_77068

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
э
G__inference_sequential_2_layer_call_and_return_conditional_losses_76456
dense_8_input
dense_8_76445
dense_8_76447
dense_9_76450
dense_9_76452
identityИҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallЫ
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_76445dense_8_76447*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€иd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_763792!
dense_8/StatefulPartitionedCallґ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_76450dense_9_76452*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_764252!
dense_9/StatefulPartitionedCall≈
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:[ W
,
_output_shapes
:€€€€€€€€€и 
'
_user_specified_namedense_8_input
н	
№
C__inference_dense_10_layer_call_and_return_conditional_losses_77011

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
вH
¶
G__inference_sequential_2_layer_call_and_return_conditional_losses_78426

inputs-
)dense_8_tensordot_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityИҐdense_8/BiasAdd/ReadVariableOpҐ dense_8/Tensordot/ReadVariableOpҐdense_9/BiasAdd/ReadVariableOpҐ dense_9/Tensordot/ReadVariableOpЃ
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axesБ
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_8/Tensordot/freeh
dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_8/Tensordot/ShapeД
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axisщ
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2И
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axis€
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const†
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/ProdА
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1®
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1А
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axisЎ
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concatђ
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stack©
dense_8/Tensordot/transpose	Transposeinputs!dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dense_8/Tensordot/transposeњ
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_8/Tensordot/ReshapeЊ
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
dense_8/Tensordot/MatMulА
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
dense_8/Tensordot/Const_2Д
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axisе
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1±
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_8/Tensordot§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_8/BiasAdd/ReadVariableOp®
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_8/BiasAddu
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_8/ReluЃ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axesБ
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/free|
dense_9/Tensordot/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:2
dense_9/Tensordot/ShapeД
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisщ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2И
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis€
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const†
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/ProdА
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1®
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1А
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisЎ
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatђ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackљ
dense_9/Tensordot/transpose	Transposedense_8/Relu:activations:0!dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_9/Tensordot/transposeњ
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_9/Tensordot/ReshapeЊ
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_9/Tensordot/MatMulА
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_2Д
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisе
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1±
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dense_9/Tensordot§
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_9/BiasAdd/ReadVariableOp®
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dense_9/BiasAddщ
IdentityIdentitydense_9/BiasAdd:output:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
Ё
}
(__inference_dense_10_layer_call_fn_78322

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_770112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ґ 
б
B__inference_dense_8_layer_call_and_return_conditional_losses_76379

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€иd2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€и ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
Ї
Я
,__inference_sequential_2_layer_call_fn_78509

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_765002
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
Ё
}
(__inference_dense_11_layer_call_fn_78369

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_770682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ч
F
*__inference_dropout_10_layer_call_fn_78302

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_769872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ч
F
*__inference_dropout_11_layer_call_fn_78349

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_770442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
£
c
*__inference_dropout_11_layer_call_fn_78344

inputs
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_770392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
С
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_78259

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
х	
№
C__inference_dense_11_layer_call_and_return_conditional_losses_78360

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
“

я
3__inference_transformer_block_2_layer_call_fn_78253

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_768492
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€и ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
ть
—
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_78052

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityИҐ.layer_normalization_4/batchnorm/ReadVariableOpҐ2layer_normalization_4/batchnorm/mul/ReadVariableOpҐ.layer_normalization_5/batchnorm/ReadVariableOpҐ2layer_normalization_5/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_2/attention_output/add/ReadVariableOpҐDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_2/key/add/ReadVariableOpҐ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/query/add/ReadVariableOpҐ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/value/add/ReadVariableOpҐ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐ+sequential_2/dense_8/BiasAdd/ReadVariableOpҐ-sequential_2/dense_8/Tensordot/ReadVariableOpҐ+sequential_2/dense_9/BiasAdd/ReadVariableOpҐ-sequential_2/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumџ
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpц
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/query/addч
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsum’
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpо
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2 
multi_head_attention_2/key/addэ
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumџ
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpц
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/value/addБ
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_2/Mul/y«
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2
multi_head_attention_2/Mulю
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/Einsum∆
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2(
&multi_head_attention_2/softmax/Softmax°
,multi_head_attention_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_2/dropout/dropout/ConstД
*multi_head_attention_2/dropout/dropout/MulMul0multi_head_attention_2/softmax/Softmax:softmax:05multi_head_attention_2/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2,
*multi_head_attention_2/dropout/dropout/MulЉ
,multi_head_attention_2/dropout/dropout/ShapeShape0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_2/dropout/dropout/ShapeЫ
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_2/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии*
dtype02E
Cmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_2/dropout/dropout/GreaterEqual/yƒ
3multi_head_attention_2/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_2/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_2/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии25
3multi_head_attention_2/dropout/dropout/GreaterEqualж
+multi_head_attention_2/dropout/dropout/CastCast7multi_head_attention_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€
ии2-
+multi_head_attention_2/dropout/dropout/CastА
,multi_head_attention_2/dropout/dropout/Mul_1Mul.multi_head_attention_2/dropout/dropout/Mul:z:0/multi_head_attention_2/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2.
,multi_head_attention_2/dropout/dropout/Mul_1Х
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/dropout/Mul_1:z:0$multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/EinsumЮ
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumш
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOpЮ
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2-
+multi_head_attention_2/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_8/dropout/Constњ
dropout_8/dropout/MulMul/multi_head_attention_2/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/dropout/MulС
dropout_8/dropout/ShapeShape/multi_head_attention_2/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape„
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€и *
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЙ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_8/dropout/GreaterEqual/yл
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
dropout_8/dropout/GreaterEqualҐ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/dropout/CastІ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/dropout/Mul_1o
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
addґ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesа
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_4/moments/meanћ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_4/moments/StopGradientм
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_4/moments/SquaredDifferenceЊ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesШ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_4/moments/varianceУ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_4/batchnorm/add/yл
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_4/batchnorm/addЈ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_4/batchnorm/Rsqrtа
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpп
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/mulЊ
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_1в
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_2‘
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpл
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/subв
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/add_1’
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOpФ
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axesЫ
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/free•
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/ShapeЮ
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisЇ
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ґ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisј
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1Ц
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Const‘
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/ProdЪ
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1№
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1Ъ
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axisЩ
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatа
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackу
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2*
(sequential_2/dense_8/Tensordot/transposeу
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_8/Tensordot/Reshapeт
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2'
%sequential_2/dense_8/Tensordot/MatMulЪ
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2(
&sequential_2/dense_8/Tensordot/Const_2Ю
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axis¶
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1е
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2 
sequential_2/dense_8/TensordotЋ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOp№
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/BiasAddЬ
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/Relu’
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOpФ
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axesЫ
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/free£
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/ShapeЮ
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisЇ
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ґ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisј
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1Ц
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Const‘
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/ProdЪ
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1№
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1Ъ
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axisЩ
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatа
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stackс
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2*
(sequential_2/dense_9/Tensordot/transposeу
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_9/Tensordot/Reshapeт
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/dense_9/Tensordot/MatMulЪ
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2Ю
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axis¶
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1е
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
sequential_2/dense_9/TensordotЋ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOp№
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
sequential_2/dense_9/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_9/dropout/Constµ
dropout_9/dropout/MulMul%sequential_2/dense_9/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/dropout/MulЗ
dropout_9/dropout/ShapeShape%sequential_2/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape„
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€и *
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЙ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_9/dropout/GreaterEqual/yл
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
dropout_9/dropout/GreaterEqualҐ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/dropout/CastІ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/dropout/Mul_1Ц
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
add_1ґ
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesв
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_5/moments/meanћ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_5/moments/StopGradientо
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_5/moments/SquaredDifferenceЊ
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indicesШ
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_5/moments/varianceУ
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_5/batchnorm/add/yл
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_5/batchnorm/addЈ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_5/batchnorm/Rsqrtа
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpп
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/mulј
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_1в
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_2‘
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpл
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/subв
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/add_1‘
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€и ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
£
c
*__inference_dropout_10_layer_call_fn_78297

inputs
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_769822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
вH
¶
G__inference_sequential_2_layer_call_and_return_conditional_losses_78483

inputs-
)dense_8_tensordot_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityИҐdense_8/BiasAdd/ReadVariableOpҐ dense_8/Tensordot/ReadVariableOpҐdense_9/BiasAdd/ReadVariableOpҐ dense_9/Tensordot/ReadVariableOpЃ
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axesБ
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_8/Tensordot/freeh
dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_8/Tensordot/ShapeД
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axisщ
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2И
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axis€
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const†
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/ProdА
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1®
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1А
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axisЎ
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concatђ
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stack©
dense_8/Tensordot/transpose	Transposeinputs!dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dense_8/Tensordot/transposeњ
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_8/Tensordot/ReshapeЊ
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
dense_8/Tensordot/MatMulА
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
dense_8/Tensordot/Const_2Д
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axisе
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1±
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_8/Tensordot§
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_8/BiasAdd/ReadVariableOp®
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_8/BiasAddu
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_8/ReluЃ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axesБ
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/free|
dense_9/Tensordot/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:2
dense_9/Tensordot/ShapeД
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisщ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2И
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis€
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const†
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/ProdА
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1®
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1А
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisЎ
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatђ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stackљ
dense_9/Tensordot/transpose	Transposedense_8/Relu:activations:0!dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
dense_9/Tensordot/transposeњ
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_9/Tensordot/ReshapeЊ
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_9/Tensordot/MatMulА
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_2Д
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisе
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1±
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dense_9/Tensordot§
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_9/BiasAdd/ReadVariableOp®
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dense_9/BiasAddщ
IdentityIdentitydense_9/BiasAdd:output:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
’
б
B__inference_dense_9_layer_call_and_return_conditional_losses_76425

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€иd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€иd
 
_user_specified_nameinputs
Ґб
Л
B__inference_model_2_layer_call_and_return_conditional_losses_77773

inputsE
Atoken_and_position_embedding_2_embedding_5_embedding_lookup_77619E
Atoken_and_position_embedding_2_embedding_4_embedding_lookup_77625Z
Vtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_query_add_readvariableop_resourceX
Ttransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resourceZ
Vtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_2_multi_head_attention_2_value_add_readvariableop_resourcee
atransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resourceS
Otransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resourceN
Jtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resourceL
Htransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resourceS
Otransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityИҐdense_10/BiasAdd/ReadVariableOpҐdense_10/MatMul/ReadVariableOpҐdense_11/BiasAdd/ReadVariableOpҐdense_11/MatMul/ReadVariableOpҐ;token_and_position_embedding_2/embedding_4/embedding_lookupҐ;token_and_position_embedding_2/embedding_5/embedding_lookupҐBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpҐFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpҐBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpҐFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpҐNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpҐXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpҐKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpҐMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpҐMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐ?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpҐAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpҐ?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpҐAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpВ
$token_and_position_embedding_2/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shapeї
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€24
2token_and_position_embedding_2/strided_slice/stackґ
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1ґ
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2Ь
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_sliceЪ
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/startЪ
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/deltaЫ
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2&
$token_and_position_embedding_2/range»
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_5_embedding_lookup_77619-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/77619*'
_output_shapes
:€€€€€€€€€ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookupФ
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/77619*'
_output_shapes
:€€€€€€€€€ 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityЭ
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1і
/token_and_position_embedding_2/embedding_4/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€и21
/token_and_position_embedding_2/embedding_4/Cast”
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_4_embedding_lookup_776253token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/77625*,
_output_shapes
:€€€€€€€€€и *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookupЩ
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/77625*,
_output_shapes
:€€€€€€€€€и 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityҐ
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1™
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2$
"token_and_position_embedding_2/addє
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02O
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpк
>transformer_block_2/multi_head_attention_2/query/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/query/einsum/EinsumЧ
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype02E
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp∆
4transformer_block_2/multi_head_attention_2/query/addAddV2Gtransformer_block_2/multi_head_attention_2/query/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 26
4transformer_block_2/multi_head_attention_2/query/add≥
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02M
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpд
<transformer_block_2/multi_head_attention_2/key/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Stransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2>
<transformer_block_2/multi_head_attention_2/key/einsum/EinsumС
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpReadVariableOpJtransformer_block_2_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02C
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpЊ
2transformer_block_2/multi_head_attention_2/key/addAddV2Etransformer_block_2/multi_head_attention_2/key/einsum/Einsum:output:0Itransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 24
2transformer_block_2/multi_head_attention_2/key/addє
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02O
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpк
>transformer_block_2/multi_head_attention_2/value/einsum/EinsumEinsum&token_and_position_embedding_2/add:z:0Utransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2@
>transformer_block_2/multi_head_attention_2/value/einsum/EinsumЧ
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpReadVariableOpLtransformer_block_2_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype02E
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp∆
4transformer_block_2/multi_head_attention_2/value/addAddV2Gtransformer_block_2/multi_head_attention_2/value/einsum/Einsum:output:0Ktransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 26
4transformer_block_2/multi_head_attention_2/value/add©
0transformer_block_2/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>22
0transformer_block_2/multi_head_attention_2/Mul/yЧ
.transformer_block_2/multi_head_attention_2/MulMul8transformer_block_2/multi_head_attention_2/query/add:z:09transformer_block_2/multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 20
.transformer_block_2/multi_head_attention_2/Mulќ
8transformer_block_2/multi_head_attention_2/einsum/EinsumEinsum6transformer_block_2/multi_head_attention_2/key/add:z:02transformer_block_2/multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2:
8transformer_block_2/multi_head_attention_2/einsum/EinsumВ
:transformer_block_2/multi_head_attention_2/softmax/SoftmaxSoftmaxAtransformer_block_2/multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2<
:transformer_block_2/multi_head_attention_2/softmax/SoftmaxИ
;transformer_block_2/multi_head_attention_2/dropout/IdentityIdentityDtransformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2=
;transformer_block_2/multi_head_attention_2/dropout/Identityе
:transformer_block_2/multi_head_attention_2/einsum_1/EinsumEinsumDtransformer_block_2/multi_head_attention_2/dropout/Identity:output:08transformer_block_2/multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2<
:transformer_block_2/multi_head_attention_2/einsum_1/EinsumЏ
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02Z
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp§
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumEinsumCtransformer_block_2/multi_head_attention_2/einsum_1/Einsum:output:0`transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe2K
Itransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsumі
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpо
?transformer_block_2/multi_head_attention_2/attention_output/addAddV2Rtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum:output:0Vtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2A
?transformer_block_2/multi_head_attention_2/attention_output/addЎ
&transformer_block_2/dropout_8/IdentityIdentityCtransformer_block_2/multi_head_attention_2/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2(
&transformer_block_2/dropout_8/IdentityЋ
transformer_block_2/addAddV2&token_and_position_embedding_2/add:z:0/transformer_block_2/dropout_8/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
transformer_block_2/addё
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indices∞
6transformer_block_2/layer_normalization_4/moments/meanMeantransformer_block_2/add:z:0Qtransformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(28
6transformer_block_2/layer_normalization_4/moments/meanИ
>transformer_block_2/layer_normalization_4/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2@
>transformer_block_2/layer_normalization_4/moments/StopGradientЉ
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add:z:0Gtransformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2E
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceж
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesи
:transformer_block_2/layer_normalization_4/moments/varianceMeanGtransformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2<
:transformer_block_2/layer_normalization_4/moments/varianceї
9transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_2/layer_normalization_4/batchnorm/add/yї
7transformer_block_2/layer_normalization_4/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_4/moments/variance:output:0Btransformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и29
7transformer_block_2/layer_normalization_4/batchnorm/addу
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2;
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtЬ
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpњ
7transformer_block_2/layer_normalization_4/batchnorm/mulMul=transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_4/batchnorm/mulО
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Multransformer_block_2/add:z:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_1≤
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_4/moments/mean:output:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Р
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpї
7transformer_block_2/layer_normalization_4/batchnorm/subSubJtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_4/batchnorm/sub≤
9transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_4/batchnorm/add_1С
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02C
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpЉ
7transformer_block_2/sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_8/Tensordot/axes√
7transformer_block_2/sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_8/Tensordot/freeб
8transformer_block_2/sequential_2/dense_8/Tensordot/ShapeShape=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/Shape∆
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisЮ
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2 
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis§
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1Њ
8transformer_block_2/sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_8/Tensordot/Const§
7transformer_block_2/sequential_2/dense_8/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_8/Tensordot/Prod¬
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_1ђ
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1¬
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisэ
9transformer_block_2/sequential_2/dense_8/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_8/Tensordot/concat∞
8transformer_block_2/sequential_2/dense_8/Tensordot/stackPack@transformer_block_2/sequential_2/dense_8/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_8/Tensordot/stack√
<transformer_block_2/sequential_2/dense_8/Tensordot/transpose	Transpose=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Btransformer_block_2/sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2>
<transformer_block_2/sequential_2/dense_8/Tensordot/transpose√
:transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_8/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Reshape¬
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_8/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2;
9transformer_block_2/sequential_2/dense_8/Tensordot/MatMul¬
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2<
:transformer_block_2/sequential_2/dense_8/Tensordot/Const_2∆
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisК
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_8/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_8/Tensordot/concat_1µ
2transformer_block_2/sequential_2/dense_8/TensordotReshapeCtransformer_block_2/sequential_2/dense_8/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd24
2transformer_block_2/sequential_2/dense_8/TensordotЗ
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02A
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpђ
0transformer_block_2/sequential_2/dense_8/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_8/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd22
0transformer_block_2/sequential_2/dense_8/BiasAddЎ
-transformer_block_2/sequential_2/dense_8/ReluRelu9transformer_block_2/sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2/
-transformer_block_2/sequential_2/dense_8/ReluС
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02C
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpЉ
7transformer_block_2/sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_2/sequential_2/dense_9/Tensordot/axes√
7transformer_block_2/sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_2/sequential_2/dense_9/Tensordot/freeя
8transformer_block_2/sequential_2/dense_9/Tensordot/ShapeShape;transformer_block_2/sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/Shape∆
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisЮ
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2 
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis§
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Ktransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1Њ
8transformer_block_2/sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_2/sequential_2/dense_9/Tensordot/Const§
7transformer_block_2/sequential_2/dense_9/Tensordot/ProdProdDtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Atransformer_block_2/sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_2/sequential_2/dense_9/Tensordot/Prod¬
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_1ђ
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1ProdFtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1¬
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisэ
9transformer_block_2/sequential_2/dense_9/Tensordot/concatConcatV2@transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0@transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Gtransformer_block_2/sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_9/Tensordot/concat∞
8transformer_block_2/sequential_2/dense_9/Tensordot/stackPack@transformer_block_2/sequential_2/dense_9/Tensordot/Prod:output:0Btransformer_block_2/sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_2/sequential_2/dense_9/Tensordot/stackЅ
<transformer_block_2/sequential_2/dense_9/Tensordot/transpose	Transpose;transformer_block_2/sequential_2/dense_8/Relu:activations:0Btransformer_block_2/sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2>
<transformer_block_2/sequential_2/dense_9/Tensordot/transpose√
:transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeReshape@transformer_block_2/sequential_2/dense_9/Tensordot/transpose:y:0Atransformer_block_2/sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Reshape¬
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMulMatMulCtransformer_block_2/sequential_2/dense_9/Tensordot/Reshape:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2;
9transformer_block_2/sequential_2/dense_9/Tensordot/MatMul¬
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_2/sequential_2/dense_9/Tensordot/Const_2∆
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisК
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Ctransformer_block_2/sequential_2/dense_9/Tensordot/Const_2:output:0Itransformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_2/sequential_2/dense_9/Tensordot/concat_1µ
2transformer_block_2/sequential_2/dense_9/TensordotReshapeCtransformer_block_2/sequential_2/dense_9/Tensordot/MatMul:product:0Dtransformer_block_2/sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 24
2transformer_block_2/sequential_2/dense_9/TensordotЗ
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpђ
0transformer_block_2/sequential_2/dense_9/BiasAddBiasAdd;transformer_block_2/sequential_2/dense_9/Tensordot:output:0Gtransformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 22
0transformer_block_2/sequential_2/dense_9/BiasAddќ
&transformer_block_2/dropout_9/IdentityIdentity9transformer_block_2/sequential_2/dense_9/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2(
&transformer_block_2/dropout_9/Identityж
transformer_block_2/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0/transformer_block_2/dropout_9/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
transformer_block_2/add_1ё
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indices≤
6transformer_block_2/layer_normalization_5/moments/meanMeantransformer_block_2/add_1:z:0Qtransformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(28
6transformer_block_2/layer_normalization_5/moments/meanИ
>transformer_block_2/layer_normalization_5/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2@
>transformer_block_2/layer_normalization_5/moments/StopGradientЊ
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add_1:z:0Gtransformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2E
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceж
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesи
:transformer_block_2/layer_normalization_5/moments/varianceMeanGtransformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2<
:transformer_block_2/layer_normalization_5/moments/varianceї
9transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_2/layer_normalization_5/batchnorm/add/yї
7transformer_block_2/layer_normalization_5/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_5/moments/variance:output:0Btransformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и29
7transformer_block_2/layer_normalization_5/batchnorm/addу
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2;
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtЬ
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpњ
7transformer_block_2/layer_normalization_5/batchnorm/mulMul=transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_5/batchnorm/mulР
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Multransformer_block_2/add_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_1≤
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_5/moments/mean:output:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Р
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpї
7transformer_block_2/layer_normalization_5/batchnorm/subSubJtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 29
7transformer_block_2/layer_normalization_5/batchnorm/sub≤
9transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2;
9transformer_block_2/layer_normalization_5/batchnorm/add_1®
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesч
global_average_pooling1d_2/MeanMean=transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
global_average_pooling1d_2/MeanТ
dropout_10/IdentityIdentity(global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_10/Identity®
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_10/MatMul/ReadVariableOp§
dense_10/MatMulMatMuldropout_10/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_10/MatMulІ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp•
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_10/ReluЕ
dropout_11/IdentityIdentitydense_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout_11/Identity®
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp§
dense_11/MatMulMatMuldropout_11/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/MatMulІ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp•
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/SoftmaxВ
IdentityIdentitydense_11/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupC^transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpC^transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpO^transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpY^transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpL^transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpD^transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpN^transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp@^transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp@^transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpB^transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2И
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2Р
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2И
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2Р
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpNtransformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp2і
Xtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_2/multi_head_attention_2/key/add/ReadVariableOpAtransformer_block_2/multi_head_attention_2/key/add/ReadVariableOp2Ъ
Ktransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpKtransformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_2/multi_head_attention_2/query/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/query/add/ReadVariableOp2Ю
Mtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_2/multi_head_attention_2/value/add/ReadVariableOpCtransformer_block_2/multi_head_attention_2/value/add/ReadVariableOp2Ю
Mtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpMtransformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2В
?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp2Ж
Atransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp2В
?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp?transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp2Ж
Atransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpAtransformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
ч
і
'__inference_model_2_layer_call_fn_77349
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_773022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€и
!
_user_specified_name	input_3
–Д
 
 __inference__wrapped_model_76344
input_3M
Imodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_76190M
Imodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_76196b
^model_2_transformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_2_multi_head_attention_2_query_add_readvariableop_resource`
\model_2_transformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resourceV
Rmodel_2_transformer_block_2_multi_head_attention_2_key_add_readvariableop_resourceb
^model_2_transformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_2_multi_head_attention_2_value_add_readvariableop_resourcem
imodel_2_transformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resourcec
_model_2_transformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource[
Wmodel_2_transformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resourceW
Smodel_2_transformer_block_2_layer_normalization_4_batchnorm_readvariableop_resourceV
Rmodel_2_transformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resourceT
Pmodel_2_transformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resourceV
Rmodel_2_transformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resourceT
Pmodel_2_transformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource[
Wmodel_2_transformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resourceW
Smodel_2_transformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource3
/model_2_dense_10_matmul_readvariableop_resource4
0model_2_dense_10_biasadd_readvariableop_resource3
/model_2_dense_11_matmul_readvariableop_resource4
0model_2_dense_11_biasadd_readvariableop_resource
identityИҐ'model_2/dense_10/BiasAdd/ReadVariableOpҐ&model_2/dense_10/MatMul/ReadVariableOpҐ'model_2/dense_11/BiasAdd/ReadVariableOpҐ&model_2/dense_11/MatMul/ReadVariableOpҐCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupҐCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupҐJmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpҐNmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpҐJmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpҐNmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpҐVmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpҐ`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐImodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpҐSmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐKmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpҐUmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐKmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpҐUmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐGmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpҐImodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpҐGmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpҐImodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpУ
,model_2/token_and_position_embedding_2/ShapeShapeinput_3*
T0*
_output_shapes
:2.
,model_2/token_and_position_embedding_2/ShapeЋ
:model_2/token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2<
:model_2/token_and_position_embedding_2/strided_slice/stack∆
<model_2/token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_2/token_and_position_embedding_2/strided_slice/stack_1∆
<model_2/token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/token_and_position_embedding_2/strided_slice/stack_2ћ
4model_2/token_and_position_embedding_2/strided_sliceStridedSlice5model_2/token_and_position_embedding_2/Shape:output:0Cmodel_2/token_and_position_embedding_2/strided_slice/stack:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_1:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/token_and_position_embedding_2/strided_slice™
2model_2/token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/token_and_position_embedding_2/range/start™
2model_2/token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/token_and_position_embedding_2/range/delta√
,model_2/token_and_position_embedding_2/rangeRange;model_2/token_and_position_embedding_2/range/start:output:0=model_2/token_and_position_embedding_2/strided_slice:output:0;model_2/token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2.
,model_2/token_and_position_embedding_2/rangeр
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherImodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_761905model_2/token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/76190*'
_output_shapes
:€€€€€€€€€ *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupі
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/76190*'
_output_shapes
:€€€€€€€€€ 2N
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identityµ
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2P
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1≈
7model_2/token_and_position_embedding_2/embedding_4/CastCastinput_3*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€и29
7model_2/token_and_position_embedding_2/embedding_4/Castы
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherImodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_76196;model_2/token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/76196*,
_output_shapes
:€€€€€€€€€и *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupє
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/76196*,
_output_shapes
:€€€€€€€€€и 2N
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityЇ
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2P
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1 
*model_2/token_and_position_embedding_2/addAddV2Wmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2,
*model_2/token_and_position_embedding_2/add—
Umodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_2_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02W
Umodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpК
Fmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/EinsumEinsum.model_2/token_and_position_embedding_2/add:z:0]model_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2H
Fmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsumѓ
Kmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_2_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype02M
Kmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpж
<model_2/transformer_block_2/multi_head_attention_2/query/addAddV2Omodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum:output:0Smodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2>
<model_2/transformer_block_2/multi_head_attention_2/query/addЋ
Smodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_block_2_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02U
Smodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpД
Dmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/EinsumEinsum.model_2/token_and_position_embedding_2/add:z:0[model_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2F
Dmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum©
Imodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_block_2_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02K
Imodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpё
:model_2/transformer_block_2/multi_head_attention_2/key/addAddV2Mmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum:output:0Qmodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2<
:model_2/transformer_block_2/multi_head_attention_2/key/add—
Umodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_2_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02W
Umodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpК
Fmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/EinsumEinsum.model_2/token_and_position_embedding_2/add:z:0]model_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2H
Fmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsumѓ
Kmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_2_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype02M
Kmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpж
<model_2/transformer_block_2/multi_head_attention_2/value/addAddV2Omodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum:output:0Smodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2>
<model_2/transformer_block_2/multi_head_attention_2/value/addє
8model_2/transformer_block_2/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2:
8model_2/transformer_block_2/multi_head_attention_2/Mul/yЈ
6model_2/transformer_block_2/multi_head_attention_2/MulMul@model_2/transformer_block_2/multi_head_attention_2/query/add:z:0Amodel_2/transformer_block_2/multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 28
6model_2/transformer_block_2/multi_head_attention_2/Mulо
@model_2/transformer_block_2/multi_head_attention_2/einsum/EinsumEinsum>model_2/transformer_block_2/multi_head_attention_2/key/add:z:0:model_2/transformer_block_2/multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2B
@model_2/transformer_block_2/multi_head_attention_2/einsum/EinsumЪ
Bmodel_2/transformer_block_2/multi_head_attention_2/softmax/SoftmaxSoftmaxImodel_2/transformer_block_2/multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2D
Bmodel_2/transformer_block_2/multi_head_attention_2/softmax/Softmax†
Cmodel_2/transformer_block_2/multi_head_attention_2/dropout/IdentityIdentityLmodel_2/transformer_block_2/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2E
Cmodel_2/transformer_block_2/multi_head_attention_2/dropout/IdentityЕ
Bmodel_2/transformer_block_2/multi_head_attention_2/einsum_1/EinsumEinsumLmodel_2/transformer_block_2/multi_head_attention_2/dropout/Identity:output:0@model_2/transformer_block_2/multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2D
Bmodel_2/transformer_block_2/multi_head_attention_2/einsum_1/Einsumт
`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_block_2_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02b
`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpƒ
Qmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/EinsumEinsumKmodel_2/transformer_block_2/multi_head_attention_2/einsum_1/Einsum:output:0hmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe2S
Qmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsumћ
Vmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_block_2_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpО
Gmodel_2/transformer_block_2/multi_head_attention_2/attention_output/addAddV2Zmodel_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum:output:0^model_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2I
Gmodel_2/transformer_block_2/multi_head_attention_2/attention_output/addр
.model_2/transformer_block_2/dropout_8/IdentityIdentityKmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 20
.model_2/transformer_block_2/dropout_8/Identityл
model_2/transformer_block_2/addAddV2.model_2/token_and_position_embedding_2/add:z:07model_2/transformer_block_2/dropout_8/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2!
model_2/transformer_block_2/addо
Pmodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indices–
>model_2/transformer_block_2/layer_normalization_4/moments/meanMean#model_2/transformer_block_2/add:z:0Ymodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2@
>model_2/transformer_block_2/layer_normalization_4/moments/mean†
Fmodel_2/transformer_block_2/layer_normalization_4/moments/StopGradientStopGradientGmodel_2/transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2H
Fmodel_2/transformer_block_2/layer_normalization_4/moments/StopGradient№
Kmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_2/add:z:0Omodel_2/transformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2M
Kmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifferenceц
Tmodel_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indicesИ
Bmodel_2/transformer_block_2/layer_normalization_4/moments/varianceMeanOmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0]model_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2D
Bmodel_2/transformer_block_2/layer_normalization_4/moments/varianceЋ
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/yџ
?model_2/transformer_block_2/layer_normalization_4/batchnorm/addAddV2Kmodel_2/transformer_block_2/layer_normalization_4/moments/variance:output:0Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/addЛ
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrtCmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/Rsqrtі
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpя
?model_2/transformer_block_2/layer_normalization_4/batchnorm/mulMulEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Vmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/mulЃ
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1Mul#model_2/transformer_block_2/add:z:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1“
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2MulGmodel_2/transformer_block_2/layer_normalization_4/moments/mean:output:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2®
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpџ
?model_2/transformer_block_2/layer_normalization_4/batchnorm/subSubRmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/sub“
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1©
Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOpRmodel_2_transformer_block_2_sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02K
Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpћ
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2A
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/axes”
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/freeщ
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ShapeShapeEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/Shape÷
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis∆
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2GatherV2Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Qmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2Џ
Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axisћ
Emodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1GatherV2Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Smodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1ќ
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/Constƒ
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ProdProdLmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2A
?model_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod“
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_1ћ
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1ProdNmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2_1:output:0Kmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2C
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1“
Fmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat/axis•
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concatConcatV2Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/free:output:0Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/axes:output:0Omodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat–
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/stackPackHmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod:output:0Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_8/Tensordot/stackг
Dmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/transpose	TransposeEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Jmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2F
Dmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/transposeг
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReshapeReshapeHmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/transpose:y:0Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2D
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Reshapeв
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/MatMulMatMulKmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Reshape:output:0Qmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2C
Amodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/MatMul“
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2D
Bmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_2÷
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis≤
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1ConcatV2Lmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/GatherV2:output:0Kmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/Const_2:output:0Qmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1’
:model_2/transformer_block_2/sequential_2/dense_8/TensordotReshapeKmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/MatMul:product:0Lmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2<
:model_2/transformer_block_2/sequential_2/dense_8/TensordotЯ
Gmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpPmodel_2_transformer_block_2_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02I
Gmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpћ
8model_2/transformer_block_2/sequential_2/dense_8/BiasAddBiasAddCmodel_2/transformer_block_2/sequential_2/dense_8/Tensordot:output:0Omodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2:
8model_2/transformer_block_2/sequential_2/dense_8/BiasAddр
5model_2/transformer_block_2/sequential_2/dense_8/ReluReluAmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd27
5model_2/transformer_block_2/sequential_2/dense_8/Relu©
Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOpRmodel_2_transformer_block_2_sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02K
Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpћ
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2A
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/axes”
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/freeч
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ShapeShapeCmodel_2/transformer_block_2/sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/Shape÷
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis∆
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2GatherV2Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Qmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2Џ
Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axisћ
Emodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1GatherV2Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Shape:output:0Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Smodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1ќ
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/Constƒ
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ProdProdLmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2A
?model_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod“
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_1ћ
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1ProdNmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2_1:output:0Kmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2C
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1“
Fmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat/axis•
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concatConcatV2Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/free:output:0Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/axes:output:0Omodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat–
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/stackPackHmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod:output:0Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2B
@model_2/transformer_block_2/sequential_2/dense_9/Tensordot/stackб
Dmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/transpose	TransposeCmodel_2/transformer_block_2/sequential_2/dense_8/Relu:activations:0Jmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2F
Dmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/transposeг
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReshapeReshapeHmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/transpose:y:0Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2D
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Reshapeв
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/MatMulMatMulKmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Reshape:output:0Qmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2C
Amodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/MatMul“
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_2÷
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis≤
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1ConcatV2Lmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/GatherV2:output:0Kmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/Const_2:output:0Qmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1’
:model_2/transformer_block_2/sequential_2/dense_9/TensordotReshapeKmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/MatMul:product:0Lmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2<
:model_2/transformer_block_2/sequential_2/dense_9/TensordotЯ
Gmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpPmodel_2_transformer_block_2_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02I
Gmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpћ
8model_2/transformer_block_2/sequential_2/dense_9/BiasAddBiasAddCmodel_2/transformer_block_2/sequential_2/dense_9/Tensordot:output:0Omodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2:
8model_2/transformer_block_2/sequential_2/dense_9/BiasAddж
.model_2/transformer_block_2/dropout_9/IdentityIdentityAmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 20
.model_2/transformer_block_2/dropout_9/IdentityЖ
!model_2/transformer_block_2/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:07model_2/transformer_block_2/dropout_9/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2#
!model_2/transformer_block_2/add_1о
Pmodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indices“
>model_2/transformer_block_2/layer_normalization_5/moments/meanMean%model_2/transformer_block_2/add_1:z:0Ymodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2@
>model_2/transformer_block_2/layer_normalization_5/moments/mean†
Fmodel_2/transformer_block_2/layer_normalization_5/moments/StopGradientStopGradientGmodel_2/transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2H
Fmodel_2/transformer_block_2/layer_normalization_5/moments/StopGradientё
Kmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_2/add_1:z:0Omodel_2/transformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2M
Kmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifferenceц
Tmodel_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indicesИ
Bmodel_2/transformer_block_2/layer_normalization_5/moments/varianceMeanOmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0]model_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2D
Bmodel_2/transformer_block_2/layer_normalization_5/moments/varianceЋ
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/yџ
?model_2/transformer_block_2/layer_normalization_5/batchnorm/addAddV2Kmodel_2/transformer_block_2/layer_normalization_5/moments/variance:output:0Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/addЛ
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrtCmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/Rsqrtі
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpя
?model_2/transformer_block_2/layer_normalization_5/batchnorm/mulMulEmodel_2/transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Vmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/mul∞
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1Mul%model_2/transformer_block_2/add_1:z:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1“
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2MulGmodel_2/transformer_block_2/layer_normalization_5/moments/mean:output:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2®
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpџ
?model_2/transformer_block_2/layer_normalization_5/batchnorm/subSubRmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/sub“
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1Є
9model_2/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_2/global_average_pooling1d_2/Mean/reduction_indicesЧ
'model_2/global_average_pooling1d_2/MeanMeanEmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0Bmodel_2/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'model_2/global_average_pooling1d_2/Mean™
model_2/dropout_10/IdentityIdentity0model_2/global_average_pooling1d_2/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model_2/dropout_10/Identityј
&model_2/dense_10/MatMul/ReadVariableOpReadVariableOp/model_2_dense_10_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&model_2/dense_10/MatMul/ReadVariableOpƒ
model_2/dense_10/MatMulMatMul$model_2/dropout_10/Identity:output:0.model_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_10/MatMulњ
'model_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_10/BiasAdd/ReadVariableOp≈
model_2/dense_10/BiasAddBiasAdd!model_2/dense_10/MatMul:product:0/model_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_10/BiasAddЛ
model_2/dense_10/ReluRelu!model_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_10/ReluЭ
model_2/dropout_11/IdentityIdentity#model_2/dense_10/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dropout_11/Identityј
&model_2/dense_11/MatMul/ReadVariableOpReadVariableOp/model_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_11/MatMul/ReadVariableOpƒ
model_2/dense_11/MatMulMatMul$model_2/dropout_11/Identity:output:0.model_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_11/MatMulњ
'model_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_11/BiasAdd/ReadVariableOp≈
model_2/dense_11/BiasAddBiasAdd!model_2/dense_11/MatMul:product:0/model_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_11/BiasAddФ
model_2/dense_11/SoftmaxSoftmax!model_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_11/SoftmaxЇ
IdentityIdentity"model_2/dense_11/Softmax:softmax:0(^model_2/dense_10/BiasAdd/ReadVariableOp'^model_2/dense_10/MatMul/ReadVariableOp(^model_2/dense_11/BiasAdd/ReadVariableOp'^model_2/dense_11/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_2/embedding_4/embedding_lookupD^model_2/token_and_position_embedding_2/embedding_5/embedding_lookupK^model_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpO^model_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpK^model_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpO^model_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpW^model_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpa^model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpT^model_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpV^model_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpV^model_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpH^model_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpJ^model_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpH^model_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpJ^model_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2R
'model_2/dense_10/BiasAdd/ReadVariableOp'model_2/dense_10/BiasAdd/ReadVariableOp2P
&model_2/dense_10/MatMul/ReadVariableOp&model_2/dense_10/MatMul/ReadVariableOp2R
'model_2/dense_11/BiasAdd/ReadVariableOp'model_2/dense_11/BiasAdd/ReadVariableOp2P
&model_2/dense_11/MatMul/ReadVariableOp&model_2/dense_11/MatMul/ReadVariableOp2К
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup2К
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup2Ш
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpJmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2†
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpNmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2Ш
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpJmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2†
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpNmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2∞
Vmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOpVmodel_2/transformer_block_2/multi_head_attention_2/attention_output/add/ReadVariableOp2ƒ
`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_block_2/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2Ц
Imodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOpImodel_2/transformer_block_2/multi_head_attention_2/key/add/ReadVariableOp2™
Smodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_block_2/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2Ъ
Kmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOpKmodel_2/transformer_block_2/multi_head_attention_2/query/add/ReadVariableOp2Ѓ
Umodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_2/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2Ъ
Kmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOpKmodel_2/transformer_block_2/multi_head_attention_2/value/add/ReadVariableOp2Ѓ
Umodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_2/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Т
Gmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOpGmodel_2/transformer_block_2/sequential_2/dense_8/BiasAdd/ReadVariableOp2Ц
Imodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOpImodel_2/transformer_block_2/sequential_2/dense_8/Tensordot/ReadVariableOp2Т
Gmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOpGmodel_2/transformer_block_2/sequential_2/dense_9/BiasAdd/ReadVariableOp2Ц
Imodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOpImodel_2/transformer_block_2/sequential_2/dense_9/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:€€€€€€€€€и
!
_user_specified_name	input_3
л
Б
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_76558
x&
"embedding_5_embedding_lookup_76545&
"embedding_4_embedding_lookup_76551
identityИҐembedding_4/embedding_lookupҐembedding_5/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
range≠
embedding_5/embedding_lookupResourceGather"embedding_5_embedding_lookup_76545range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_5/embedding_lookup/76545*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_5/embedding_lookupШ
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_5/embedding_lookup/76545*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_5/embedding_lookup/Identityј
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_5/embedding_lookup/Identity_1q
embedding_4/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€и2
embedding_4/CastЄ
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_76551embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/76551*,
_output_shapes
:€€€€€€€€€и *
dtype02
embedding_4/embedding_lookupЭ
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/76551*,
_output_shapes
:€€€€€€€€€и 2'
%embedding_4/embedding_lookup/Identity≈
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2)
'embedding_4/embedding_lookup/Identity_1Ѓ
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
addЮ
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€и

_user_specified_namex
Б
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_77039

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
З
О
>__inference_token_and_position_embedding_2_layer_call_fn_77904
x
unknown
	unknown_0
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_765582
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:€€€€€€€€€и

_user_specified_namex
К-
м
B__inference_model_2_layer_call_and_return_conditional_losses_77198

inputs(
$token_and_position_embedding_2_77146(
$token_and_position_embedding_2_77148
transformer_block_2_77151
transformer_block_2_77153
transformer_block_2_77155
transformer_block_2_77157
transformer_block_2_77159
transformer_block_2_77161
transformer_block_2_77163
transformer_block_2_77165
transformer_block_2_77167
transformer_block_2_77169
transformer_block_2_77171
transformer_block_2_77173
transformer_block_2_77175
transformer_block_2_77177
transformer_block_2_77179
transformer_block_2_77181
dense_10_77186
dense_10_77188
dense_11_77192
dense_11_77194
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ"dropout_10/StatefulPartitionedCallҐ"dropout_11/StatefulPartitionedCallҐ6token_and_position_embedding_2/StatefulPartitionedCallҐ+transformer_block_2/StatefulPartitionedCallЗ
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_2_77146$token_and_position_embedding_2_77148*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_7655828
6token_and_position_embedding_2/StatefulPartitionedCallЯ
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_77151transformer_block_2_77153transformer_block_2_77155transformer_block_2_77157transformer_block_2_77159transformer_block_2_77161transformer_block_2_77163transformer_block_2_77165transformer_block_2_77167transformer_block_2_77169transformer_block_2_77171transformer_block_2_77173transformer_block_2_77175transformer_block_2_77177transformer_block_2_77179transformer_block_2_77181*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_767222-
+transformer_block_2/StatefulPartitionedCallЇ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_769632,
*global_average_pooling1d_2/PartitionedCall°
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_769822$
"dropout_10/StatefulPartitionedCallє
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_10_77186dense_10_77188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_770112"
 dense_10/StatefulPartitionedCallЉ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_770392$
"dropout_11/StatefulPartitionedCallє
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_11_77192dense_11_77194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_770682"
 dense_11/StatefulPartitionedCallф
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
»
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_77044

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф
≥
'__inference_model_2_layer_call_fn_77871

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_773022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
—
∞
#__inference_signature_wrapper_77408
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_763442
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€и
!
_user_specified_name	input_3
»
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_76987

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
э)
Ґ
B__inference_model_2_layer_call_and_return_conditional_losses_77302

inputs(
$token_and_position_embedding_2_77250(
$token_and_position_embedding_2_77252
transformer_block_2_77255
transformer_block_2_77257
transformer_block_2_77259
transformer_block_2_77261
transformer_block_2_77263
transformer_block_2_77265
transformer_block_2_77267
transformer_block_2_77269
transformer_block_2_77271
transformer_block_2_77273
transformer_block_2_77275
transformer_block_2_77277
transformer_block_2_77279
transformer_block_2_77281
transformer_block_2_77283
transformer_block_2_77285
dense_10_77290
dense_10_77292
dense_11_77296
dense_11_77298
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ6token_and_position_embedding_2/StatefulPartitionedCallҐ+transformer_block_2/StatefulPartitionedCallЗ
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_2_77250$token_and_position_embedding_2_77252*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_7655828
6token_and_position_embedding_2/StatefulPartitionedCallЯ
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_77255transformer_block_2_77257transformer_block_2_77259transformer_block_2_77261transformer_block_2_77263transformer_block_2_77265transformer_block_2_77267transformer_block_2_77269transformer_block_2_77271transformer_block_2_77273transformer_block_2_77275transformer_block_2_77277transformer_block_2_77279transformer_block_2_77281transformer_block_2_77283transformer_block_2_77285*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_768492-
+transformer_block_2/StatefulPartitionedCallЇ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_769632,
*global_average_pooling1d_2/PartitionedCallЙ
dropout_10/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_769872
dropout_10/PartitionedCall±
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_10_77290dense_10_77292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_770112"
 dense_10/StatefulPartitionedCall€
dropout_11/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_770442
dropout_11/PartitionedCall±
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_11_77296dense_11_77298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_770682"
 dense_11/StatefulPartitionedCall™
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
ф
≥
'__inference_model_2_layer_call_fn_77822

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_771982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
п
|
'__inference_dense_9_layer_call_fn_78588

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_764252
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€иd::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€иd
 
_user_specified_nameinputs
н	
№
C__inference_dense_10_layer_call_and_return_conditional_losses_78313

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ
V
:__inference_global_average_pooling1d_2_layer_call_fn_78275

inputs
identity÷
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_769632
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€и :T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
Ћ
ц
G__inference_sequential_2_layer_call_and_return_conditional_losses_76473

inputs
dense_8_76462
dense_8_76464
dense_9_76467
dense_9_76469
identityИҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallФ
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_76462dense_8_76464*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€иd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_763792!
dense_8/StatefulPartitionedCallґ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_76467dense_9_76469*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_764252!
dense_9/StatefulPartitionedCall≈
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
і№
—
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_76849

inputsF
Bmulti_head_attention_2_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_query_add_readvariableop_resourceD
@multi_head_attention_2_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_2_key_add_readvariableop_resourceF
Bmulti_head_attention_2_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_2_value_add_readvariableop_resourceQ
Mmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_2_attention_output_add_readvariableop_resource?
;layer_normalization_4_batchnorm_mul_readvariableop_resource;
7layer_normalization_4_batchnorm_readvariableop_resource:
6sequential_2_dense_8_tensordot_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource:
6sequential_2_dense_9_tensordot_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource?
;layer_normalization_5_batchnorm_mul_readvariableop_resource;
7layer_normalization_5_batchnorm_readvariableop_resource
identityИҐ.layer_normalization_4/batchnorm/ReadVariableOpҐ2layer_normalization_4/batchnorm/mul/ReadVariableOpҐ.layer_normalization_5/batchnorm/ReadVariableOpҐ2layer_normalization_5/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_2/attention_output/add/ReadVariableOpҐDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_2/key/add/ReadVariableOpҐ7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/query/add/ReadVariableOpҐ9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_2/value/add/ReadVariableOpҐ9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpҐ+sequential_2/dense_8/BiasAdd/ReadVariableOpҐ-sequential_2/dense_8/Tensordot/ReadVariableOpҐ+sequential_2/dense_9/BiasAdd/ReadVariableOpҐ-sequential_2/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/query/einsum/EinsumEinsuminputsAmulti_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/query/einsum/Einsumџ
/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp8multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/query/add/ReadVariableOpц
 multi_head_attention_2/query/addAddV23multi_head_attention_2/query/einsum/Einsum:output:07multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/query/addч
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype029
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_2/key/einsum/EinsumEinsuminputs?multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2*
(multi_head_attention_2/key/einsum/Einsum’
-multi_head_attention_2/key/add/ReadVariableOpReadVariableOp6multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:
 *
dtype02/
-multi_head_attention_2/key/add/ReadVariableOpо
multi_head_attention_2/key/addAddV21multi_head_attention_2/key/einsum/Einsum:output:05multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2 
multi_head_attention_2/key/addэ
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: 
 *
dtype02;
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_2/value/einsum/EinsumEinsuminputsAmulti_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationabc,cde->abde2,
*multi_head_attention_2/value/einsum/Einsumџ
/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp8multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:
 *
dtype021
/multi_head_attention_2/value/add/ReadVariableOpц
 multi_head_attention_2/value/addAddV23multi_head_attention_2/value/einsum/Einsum:output:07multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2"
 multi_head_attention_2/value/addБ
multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_2/Mul/y«
multi_head_attention_2/MulMul$multi_head_attention_2/query/add:z:0%multi_head_attention_2/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€и
 2
multi_head_attention_2/Mulю
$multi_head_attention_2/einsum/EinsumEinsum"multi_head_attention_2/key/add:z:0multi_head_attention_2/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€
ии*
equationaecd,abcd->acbe2&
$multi_head_attention_2/einsum/Einsum∆
&multi_head_attention_2/softmax/SoftmaxSoftmax-multi_head_attention_2/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2(
&multi_head_attention_2/softmax/Softmaxћ
'multi_head_attention_2/dropout/IdentityIdentity0multi_head_attention_2/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€
ии2)
'multi_head_attention_2/dropout/IdentityХ
&multi_head_attention_2/einsum_1/EinsumEinsum0multi_head_attention_2/dropout/Identity:output:0$multi_head_attention_2/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€и
 *
equationacbe,aecd->abcd2(
&multi_head_attention_2/einsum_1/EinsumЮ
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:
  *
dtype02F
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_2/attention_output/einsum/EinsumEinsum/multi_head_attention_2/einsum_1/Einsum:output:0Lmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€и *
equationabcd,cde->abe27
5multi_head_attention_2/attention_output/einsum/Einsumш
:multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_2/attention_output/add/ReadVariableOpЮ
+multi_head_attention_2/attention_output/addAddV2>multi_head_attention_2/attention_output/einsum/Einsum:output:0Bmulti_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2-
+multi_head_attention_2/attention_output/addЬ
dropout_8/IdentityIdentity/multi_head_attention_2/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_8/Identityo
addAddV2inputsdropout_8/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
addґ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesа
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_4/moments/meanћ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_4/moments/StopGradientм
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_4/moments/SquaredDifferenceЊ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesШ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_4/moments/varianceУ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_4/batchnorm/add/yл
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_4/batchnorm/addЈ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_4/batchnorm/Rsqrtа
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpп
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/mulЊ
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_1в
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/mul_2‘
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpл
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_4/batchnorm/subв
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_4/batchnorm/add_1’
-sequential_2/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_8_tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02/
-sequential_2/dense_8/Tensordot/ReadVariableOpФ
#sequential_2/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_8/Tensordot/axesЫ
#sequential_2/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_8/Tensordot/free•
$sequential_2/dense_8/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/ShapeЮ
,sequential_2/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/GatherV2/axisЇ
'sequential_2/dense_8/Tensordot/GatherV2GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/free:output:05sequential_2/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/GatherV2Ґ
.sequential_2/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_8/Tensordot/GatherV2_1/axisј
)sequential_2/dense_8/Tensordot/GatherV2_1GatherV2-sequential_2/dense_8/Tensordot/Shape:output:0,sequential_2/dense_8/Tensordot/axes:output:07sequential_2/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_8/Tensordot/GatherV2_1Ц
$sequential_2/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_8/Tensordot/Const‘
#sequential_2/dense_8/Tensordot/ProdProd0sequential_2/dense_8/Tensordot/GatherV2:output:0-sequential_2/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_8/Tensordot/ProdЪ
&sequential_2/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_8/Tensordot/Const_1№
%sequential_2/dense_8/Tensordot/Prod_1Prod2sequential_2/dense_8/Tensordot/GatherV2_1:output:0/sequential_2/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_8/Tensordot/Prod_1Ъ
*sequential_2/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_8/Tensordot/concat/axisЩ
%sequential_2/dense_8/Tensordot/concatConcatV2,sequential_2/dense_8/Tensordot/free:output:0,sequential_2/dense_8/Tensordot/axes:output:03sequential_2/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_8/Tensordot/concatа
$sequential_2/dense_8/Tensordot/stackPack,sequential_2/dense_8/Tensordot/Prod:output:0.sequential_2/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_8/Tensordot/stackу
(sequential_2/dense_8/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0.sequential_2/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2*
(sequential_2/dense_8/Tensordot/transposeу
&sequential_2/dense_8/Tensordot/ReshapeReshape,sequential_2/dense_8/Tensordot/transpose:y:0-sequential_2/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_8/Tensordot/Reshapeт
%sequential_2/dense_8/Tensordot/MatMulMatMul/sequential_2/dense_8/Tensordot/Reshape:output:05sequential_2/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2'
%sequential_2/dense_8/Tensordot/MatMulЪ
&sequential_2/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2(
&sequential_2/dense_8/Tensordot/Const_2Ю
,sequential_2/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_8/Tensordot/concat_1/axis¶
'sequential_2/dense_8/Tensordot/concat_1ConcatV20sequential_2/dense_8/Tensordot/GatherV2:output:0/sequential_2/dense_8/Tensordot/Const_2:output:05sequential_2/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_8/Tensordot/concat_1е
sequential_2/dense_8/TensordotReshape/sequential_2/dense_8/Tensordot/MatMul:product:00sequential_2/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2 
sequential_2/dense_8/TensordotЋ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOp№
sequential_2/dense_8/BiasAddBiasAdd'sequential_2/dense_8/Tensordot:output:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/BiasAddЬ
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
sequential_2/dense_8/Relu’
-sequential_2/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_9_tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02/
-sequential_2/dense_9/Tensordot/ReadVariableOpФ
#sequential_2/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_2/dense_9/Tensordot/axesЫ
#sequential_2/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_2/dense_9/Tensordot/free£
$sequential_2/dense_9/Tensordot/ShapeShape'sequential_2/dense_8/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/ShapeЮ
,sequential_2/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/GatherV2/axisЇ
'sequential_2/dense_9/Tensordot/GatherV2GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/free:output:05sequential_2/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/GatherV2Ґ
.sequential_2/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_2/dense_9/Tensordot/GatherV2_1/axisј
)sequential_2/dense_9/Tensordot/GatherV2_1GatherV2-sequential_2/dense_9/Tensordot/Shape:output:0,sequential_2/dense_9/Tensordot/axes:output:07sequential_2/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_2/dense_9/Tensordot/GatherV2_1Ц
$sequential_2/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/dense_9/Tensordot/Const‘
#sequential_2/dense_9/Tensordot/ProdProd0sequential_2/dense_9/Tensordot/GatherV2:output:0-sequential_2/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_2/dense_9/Tensordot/ProdЪ
&sequential_2/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_1№
%sequential_2/dense_9/Tensordot/Prod_1Prod2sequential_2/dense_9/Tensordot/GatherV2_1:output:0/sequential_2/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_2/dense_9/Tensordot/Prod_1Ъ
*sequential_2/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/dense_9/Tensordot/concat/axisЩ
%sequential_2/dense_9/Tensordot/concatConcatV2,sequential_2/dense_9/Tensordot/free:output:0,sequential_2/dense_9/Tensordot/axes:output:03sequential_2/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_9/Tensordot/concatа
$sequential_2/dense_9/Tensordot/stackPack,sequential_2/dense_9/Tensordot/Prod:output:0.sequential_2/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/dense_9/Tensordot/stackс
(sequential_2/dense_9/Tensordot/transpose	Transpose'sequential_2/dense_8/Relu:activations:0.sequential_2/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2*
(sequential_2/dense_9/Tensordot/transposeу
&sequential_2/dense_9/Tensordot/ReshapeReshape,sequential_2/dense_9/Tensordot/transpose:y:0-sequential_2/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_2/dense_9/Tensordot/Reshapeт
%sequential_2/dense_9/Tensordot/MatMulMatMul/sequential_2/dense_9/Tensordot/Reshape:output:05sequential_2/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_2/dense_9/Tensordot/MatMulЪ
&sequential_2/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/dense_9/Tensordot/Const_2Ю
,sequential_2/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/dense_9/Tensordot/concat_1/axis¶
'sequential_2/dense_9/Tensordot/concat_1ConcatV20sequential_2/dense_9/Tensordot/GatherV2:output:0/sequential_2/dense_9/Tensordot/Const_2:output:05sequential_2/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_2/dense_9/Tensordot/concat_1е
sequential_2/dense_9/TensordotReshape/sequential_2/dense_9/Tensordot/MatMul:product:00sequential_2/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2 
sequential_2/dense_9/TensordotЋ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOp№
sequential_2/dense_9/BiasAddBiasAdd'sequential_2/dense_9/Tensordot:output:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
sequential_2/dense_9/BiasAddТ
dropout_9/IdentityIdentity%sequential_2/dense_9/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
dropout_9/IdentityЦ
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
add_1ґ
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indicesв
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2$
"layer_normalization_5/moments/meanћ
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2,
*layer_normalization_5/moments/StopGradientо
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 21
/layer_normalization_5/moments/SquaredDifferenceЊ
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indicesШ
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€и*
	keep_dims(2(
&layer_normalization_5/moments/varianceУ
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_5/batchnorm/add/yл
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€и2%
#layer_normalization_5/batchnorm/addЈ
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€и2'
%layer_normalization_5/batchnorm/Rsqrtа
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOpп
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/mulј
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_1в
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/mul_2‘
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOpл
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2%
#layer_normalization_5/batchnorm/subв
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€и 2'
%layer_normalization_5/batchnorm/add_1‘
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp;^multi_head_attention_2/attention_output/add/ReadVariableOpE^multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_2/key/add/ReadVariableOp8^multi_head_attention_2/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/query/add/ReadVariableOp:^multi_head_attention_2/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_2/value/add/ReadVariableOp:^multi_head_attention_2/value/einsum/Einsum/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp.^sequential_2/dense_8/Tensordot/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp.^sequential_2/dense_9/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€и ::::::::::::::::2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_2/attention_output/add/ReadVariableOp:multi_head_attention_2/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_2/key/add/ReadVariableOp-multi_head_attention_2/key/add/ReadVariableOp2r
7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp7multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/query/add/ReadVariableOp/multi_head_attention_2/query/add/ReadVariableOp2v
9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp9multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_2/value/add/ReadVariableOp/multi_head_attention_2/value/add/ReadVariableOp2v
9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp9multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2^
-sequential_2/dense_8/Tensordot/ReadVariableOp-sequential_2/dense_8/Tensordot/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2^
-sequential_2/dense_9/Tensordot/ReadVariableOp-sequential_2/dense_9/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
ґ 
б
B__inference_dense_8_layer_call_and_return_conditional_losses_78540

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: d*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€иd2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€иd2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€и ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
’
б
B__inference_dense_9_layer_call_and_return_conditional_losses_78579

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€иd2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€и 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€и 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€иd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€иd
 
_user_specified_nameinputs
¶§
У+
__inference__traced_save_78836
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopV
Rsavev2_transformer_block_2_multi_head_attention_2_query_kernel_read_readvariableopT
Psavev2_transformer_block_2_multi_head_attention_2_query_bias_read_readvariableopT
Psavev2_transformer_block_2_multi_head_attention_2_key_kernel_read_readvariableopR
Nsavev2_transformer_block_2_multi_head_attention_2_key_bias_read_readvariableopV
Rsavev2_transformer_block_2_multi_head_attention_2_value_kernel_read_readvariableopT
Psavev2_transformer_block_2_multi_head_attention_2_value_bias_read_readvariableopa
]savev2_transformer_block_2_multi_head_attention_2_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_2_multi_head_attention_2_attention_output_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableopN
Jsavev2_transformer_block_2_layer_normalization_4_gamma_read_readvariableopM
Isavev2_transformer_block_2_layer_normalization_4_beta_read_readvariableopN
Jsavev2_transformer_block_2_layer_normalization_5_gamma_read_readvariableopM
Isavev2_transformer_block_2_layer_normalization_5_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_m_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_m_read_readvariableopY
Usavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_m_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_4_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_4_beta_m_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_5_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_5_beta_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_v_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_v_read_readvariableopY
Usavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_v_read_readvariableop]
Ysavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_4_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_4_beta_v_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_5_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_5_beta_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename $
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*№#
value“#Bѕ#LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names£
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*≠
value£B†LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesБ*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopRsavev2_transformer_block_2_multi_head_attention_2_query_kernel_read_readvariableopPsavev2_transformer_block_2_multi_head_attention_2_query_bias_read_readvariableopPsavev2_transformer_block_2_multi_head_attention_2_key_kernel_read_readvariableopNsavev2_transformer_block_2_multi_head_attention_2_key_bias_read_readvariableopRsavev2_transformer_block_2_multi_head_attention_2_value_kernel_read_readvariableopPsavev2_transformer_block_2_multi_head_attention_2_value_bias_read_readvariableop]savev2_transformer_block_2_multi_head_attention_2_attention_output_kernel_read_readvariableop[savev2_transformer_block_2_multi_head_attention_2_attention_output_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableopJsavev2_transformer_block_2_layer_normalization_4_gamma_read_readvariableopIsavev2_transformer_block_2_layer_normalization_4_beta_read_readvariableopJsavev2_transformer_block_2_layer_normalization_5_gamma_read_readvariableopIsavev2_transformer_block_2_layer_normalization_5_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_m_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_m_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_m_read_readvariableopUsavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_m_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_m_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_m_read_readvariableopdsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_m_read_readvariableopbsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_4_gamma_m_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_4_beta_m_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_5_gamma_m_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_5_beta_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_query_kernel_v_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_query_bias_v_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_key_kernel_v_read_readvariableopUsavev2_adam_transformer_block_2_multi_head_attention_2_key_bias_v_read_readvariableopYsavev2_adam_transformer_block_2_multi_head_attention_2_value_kernel_v_read_readvariableopWsavev2_adam_transformer_block_2_multi_head_attention_2_value_bias_v_read_readvariableopdsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_kernel_v_read_readvariableopbsavev2_adam_transformer_block_2_multi_head_attention_2_attention_output_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_4_gamma_v_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_4_beta_v_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_5_gamma_v_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_5_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*М
_input_shapesъ
ч: : :::: : : : : :
Ні :	и : 
 :
 : 
 :
 : 
 :
 :
  : : d:d:d : : : : : : : : : : ::::
Ні :	и : 
 :
 : 
 :
 : 
 :
 :
  : : d:d:d : : : : : : ::::
Ні :	и : 
 :
 : 
 :
 : 
 :
 :
  : : d:d:d : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :&
"
 
_output_shapes
:
Ні :%!

_output_shapes
:	и :($
"
_output_shapes
: 
 :$ 

_output_shapes

:
 :($
"
_output_shapes
: 
 :$ 

_output_shapes

:
 :($
"
_output_shapes
: 
 :$ 

_output_shapes

:
 :($
"
_output_shapes
:
  : 

_output_shapes
: :$ 

_output_shapes

: d: 

_output_shapes
:d:$ 

_output_shapes

:d : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::&$"
 
_output_shapes
:
Ні :%%!

_output_shapes
:	и :(&$
"
_output_shapes
: 
 :$' 

_output_shapes

:
 :(($
"
_output_shapes
: 
 :$) 

_output_shapes

:
 :(*$
"
_output_shapes
: 
 :$+ 

_output_shapes

:
 :(,$
"
_output_shapes
:
  : -

_output_shapes
: :$. 

_output_shapes

: d: /

_output_shapes
:d:$0 

_output_shapes

:d : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: :$6 

_output_shapes

: : 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::&:"
 
_output_shapes
:
Ні :%;!

_output_shapes
:	и :(<$
"
_output_shapes
: 
 :$= 

_output_shapes

:
 :(>$
"
_output_shapes
: 
 :$? 

_output_shapes

:
 :(@$
"
_output_shapes
: 
 :$A 

_output_shapes

:
 :(B$
"
_output_shapes
:
  : C

_output_shapes
: :$D 

_output_shapes

: d: E

_output_shapes
:d:$F 

_output_shapes

:d : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :L

_output_shapes
: 
ч
і
'__inference_model_2_layer_call_fn_77245
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_771982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€и
!
_user_specified_name	input_3
х
V
:__inference_global_average_pooling1d_2_layer_call_fn_78264

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_765272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
а
э
G__inference_sequential_2_layer_call_and_return_conditional_losses_76442
dense_8_input
dense_8_76390
dense_8_76392
dense_9_76436
dense_9_76438
identityИҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallЫ
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_76390dense_8_76392*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€иd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_763792!
dense_8/StatefulPartitionedCallґ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_76436dense_9_76438*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_764252!
dense_9/StatefulPartitionedCall≈
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:[ W
,
_output_shapes
:€€€€€€€€€и 
'
_user_specified_namedense_8_input
Ё
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_78270

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€и :T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs
Н-
н
B__inference_model_2_layer_call_and_return_conditional_losses_77085
input_3(
$token_and_position_embedding_2_76569(
$token_and_position_embedding_2_76571
transformer_block_2_76925
transformer_block_2_76927
transformer_block_2_76929
transformer_block_2_76931
transformer_block_2_76933
transformer_block_2_76935
transformer_block_2_76937
transformer_block_2_76939
transformer_block_2_76941
transformer_block_2_76943
transformer_block_2_76945
transformer_block_2_76947
transformer_block_2_76949
transformer_block_2_76951
transformer_block_2_76953
transformer_block_2_76955
dense_10_77022
dense_10_77024
dense_11_77079
dense_11_77081
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ"dropout_10/StatefulPartitionedCallҐ"dropout_11/StatefulPartitionedCallҐ6token_and_position_embedding_2/StatefulPartitionedCallҐ+transformer_block_2/StatefulPartitionedCallИ
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3$token_and_position_embedding_2_76569$token_and_position_embedding_2_76571*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_7655828
6token_and_position_embedding_2/StatefulPartitionedCallЯ
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_76925transformer_block_2_76927transformer_block_2_76929transformer_block_2_76931transformer_block_2_76933transformer_block_2_76935transformer_block_2_76937transformer_block_2_76939transformer_block_2_76941transformer_block_2_76943transformer_block_2_76945transformer_block_2_76947transformer_block_2_76949transformer_block_2_76951transformer_block_2_76953transformer_block_2_76955*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_767222-
+transformer_block_2/StatefulPartitionedCallЇ
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_769632,
*global_average_pooling1d_2/PartitionedCall°
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_769822$
"dropout_10/StatefulPartitionedCallє
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_10_77022dense_10_77024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_770112"
 dense_10/StatefulPartitionedCallЉ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_770392$
"dropout_11/StatefulPartitionedCallє
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_11_77079dense_11_77081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_770682"
 dense_11/StatefulPartitionedCallф
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€и::::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€и
!
_user_specified_name	input_3
С
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_76527

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ
ц
G__inference_sequential_2_layer_call_and_return_conditional_losses_76500

inputs
dense_8_76489
dense_8_76491
dense_9_76494
dense_9_76496
identityИҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallФ
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_76489dense_8_76491*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€иd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_763792!
dense_8/StatefulPartitionedCallґ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_76494dense_9_76496*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€и *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_764252!
dense_9/StatefulPartitionedCall≈
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€и 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€и ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€и 
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ђ
serving_defaultШ
<
input_31
serving_default_input_3:0€€€€€€€€€и<
dense_110
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:“щ
¶
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
+ѕ&call_and_return_all_conditional_losses
–__call__
—_default_save_signature"И
_tf_keras_networkм{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_2", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d_2", "inbound_nodes": [[["transformer_block_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["global_average_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1000]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
п"м
_tf_keras_input_layerћ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
з
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
+“&call_and_return_all_conditional_losses
”__call__"Ї
_tf_keras_layer†{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Д
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
trainable_variables
regularization_losses
	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"•
_tf_keras_layerЛ{"class_name": "TransformerBlock", "name": "transformer_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Щ
	variables
 trainable_variables
!regularization_losses
"	keras_api
+÷&call_and_return_all_conditional_losses
„__call__"И
_tf_keras_layerо{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
й
#	variables
$trainable_variables
%regularization_losses
&	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ф

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
+Џ&call_and_return_all_conditional_losses
џ__call__"Ќ
_tf_keras_layer≥{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
й
-	variables
.trainable_variables
/regularization_losses
0	keras_api
+№&call_and_return_all_conditional_losses
Ё__call__"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ц

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+ё&call_and_return_all_conditional_losses
я__call__"ѕ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
Л
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm£(m§1m•2m¶<mІ=m®>m©?m™@mЂAmђBm≠CmЃDmѓEm∞Fm±Gm≤Hm≥ImіJmµKmґLmЈMmЄ'vє(vЇ1vї2vЉ<vљ=vЊ>vњ?vј@vЅAv¬Bv√CvƒDv≈Ev∆Fv«Gv»Hv…Iv JvЋKvћLvЌMvќ"
	optimizer
∆
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
∆
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
Nnon_trainable_variables

	variables
trainable_variables

Olayers
regularization_losses
Player_regularization_losses
Qmetrics
Rlayer_metrics
–__call__
—_default_save_signature
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
-
аserving_default"
signature_map
≥
<
embeddings
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Т
_tf_keras_layerш{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 23053, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
ђ
=
embeddings
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
+г&call_and_return_all_conditional_losses
д__call__"Л
_tf_keras_layerс{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1000, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
[non_trainable_variables
	variables
trainable_variables

\layers
regularization_losses
]layer_regularization_losses
^metrics
_layer_metrics
”__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
€
`_query_dense
a
_key_dense
b_value_dense
c_softmax
d_dropout_layer
e_output_dense
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"Е
_tf_keras_layerл{"class_name": "MultiHeadAttention", "name": "multi_head_attention_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_2", "trainable": true, "dtype": "float32", "num_heads": 10, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
©
jlayer_with_weights-0
jlayer-0
klayer_with_weights-1
klayer-1
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
+з&call_and_return_all_conditional_losses
и__call__" 
_tf_keras_sequentialЂ{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
е
paxis
	Jgamma
Kbeta
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
+й&call_and_return_all_conditional_losses
к__call__"µ
_tf_keras_layerЫ{"class_name": "LayerNormalization", "name": "layer_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}}
е
uaxis
	Lgamma
Mbeta
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+л&call_and_return_all_conditional_losses
м__call__"µ
_tf_keras_layerЫ{"class_name": "LayerNormalization", "name": "layer_normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}}
з
z	variables
{trainable_variables
|regularization_losses
}	keras_api
+н&call_and_return_all_conditional_losses
о__call__"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
й
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
+п&call_and_return_all_conditional_losses
р__call__"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Ц
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
Ц
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Вnon_trainable_variables
	variables
trainable_variables
Гlayers
regularization_losses
 Дlayer_regularization_losses
Еmetrics
Жlayer_metrics
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Зnon_trainable_variables
	variables
 trainable_variables
Иlayers
!regularization_losses
 Йlayer_regularization_losses
Кmetrics
Лlayer_metrics
„__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Мnon_trainable_variables
#	variables
$trainable_variables
Нlayers
%regularization_losses
 Оlayer_regularization_losses
Пmetrics
Рlayer_metrics
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_10/kernel
:2dense_10/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Сnon_trainable_variables
)	variables
*trainable_variables
Тlayers
+regularization_losses
 Уlayer_regularization_losses
Фmetrics
Хlayer_metrics
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Цnon_trainable_variables
-	variables
.trainable_variables
Чlayers
/regularization_losses
 Шlayer_regularization_losses
Щmetrics
Ъlayer_metrics
Ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
!:2dense_11/kernel
:2dense_11/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ыnon_trainable_variables
3	variables
4trainable_variables
Ьlayers
5regularization_losses
 Эlayer_regularization_losses
Юmetrics
Яlayer_metrics
я__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
I:G
Ні 25token_and_position_embedding_2/embedding_4/embeddings
H:F	и 25token_and_position_embedding_2/embedding_5/embeddings
M:K 
 27transformer_block_2/multi_head_attention_2/query/kernel
G:E
 25transformer_block_2/multi_head_attention_2/query/bias
K:I 
 25transformer_block_2/multi_head_attention_2/key/kernel
E:C
 23transformer_block_2/multi_head_attention_2/key/bias
M:K 
 27transformer_block_2/multi_head_attention_2/value/kernel
G:E
 25transformer_block_2/multi_head_attention_2/value/bias
X:V
  2Btransformer_block_2/multi_head_attention_2/attention_output/kernel
N:L 2@transformer_block_2/multi_head_attention_2/attention_output/bias
 : d2dense_8/kernel
:d2dense_8/bias
 :d 2dense_9/kernel
: 2dense_9/bias
=:; 2/transformer_block_2/layer_normalization_4/gamma
<:: 2.transformer_block_2/layer_normalization_4/beta
=:; 2/transformer_block_2/layer_normalization_5/gamma
<:: 2.transformer_block_2/layer_normalization_5/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
0
†0
°1"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ґnon_trainable_variables
S	variables
Ttrainable_variables
£layers
Uregularization_losses
 §layer_regularization_losses
•metrics
¶layer_metrics
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
'
=0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Іnon_trainable_variables
W	variables
Xtrainable_variables
®layers
Yregularization_losses
 ©layer_regularization_losses
™metrics
Ђlayer_metrics
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћ
ђpartial_output_shape
≠full_output_shape

>kernel
?bias
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
+с&call_and_return_all_conditional_losses
т__call__"о
_tf_keras_layer‘{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 10, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}}
»
≤partial_output_shape
≥full_output_shape

@kernel
Abias
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"к
_tf_keras_layer–{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 10, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}}
ћ
Єpartial_output_shape
єfull_output_shape

Bkernel
Cbias
Ї	variables
їtrainable_variables
Љregularization_losses
љ	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"о
_tf_keras_layer‘{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 10, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}}
л
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"÷
_tf_keras_layerЉ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
з
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"“
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
б
∆partial_output_shape
«full_output_shape

Dkernel
Ebias
»	variables
…trainable_variables
 regularization_losses
Ћ	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"Г
_tf_keras_layerй{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 10, 32]}}
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ћnon_trainable_variables
f	variables
gtrainable_variables
Ќlayers
hregularization_losses
 ќlayer_regularization_losses
ѕmetrics
–layer_metrics
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
э

Fkernel
Gbias
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"“
_tf_keras_layerЄ{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 32]}}
А

Hkernel
Ibias
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
+€&call_and_return_all_conditional_losses
А__call__"’
_tf_keras_layerї{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 100]}}
<
F0
G1
H2
I3"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ўnon_trainable_variables
l	variables
mtrainable_variables
Џlayers
nregularization_losses
 џlayer_regularization_losses
№metrics
Ёlayer_metrics
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ёnon_trainable_variables
q	variables
rtrainable_variables
яlayers
sregularization_losses
 аlayer_regularization_losses
бmetrics
вlayer_metrics
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
гnon_trainable_variables
v	variables
wtrainable_variables
дlayers
xregularization_losses
 еlayer_regularization_losses
жmetrics
зlayer_metrics
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
иnon_trainable_variables
z	variables
{trainable_variables
йlayers
|regularization_losses
 кlayer_regularization_losses
лmetrics
мlayer_metrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
нnon_trainable_variables
~	variables
trainable_variables
оlayers
Аregularization_losses
 пlayer_regularization_losses
рmetrics
сlayer_metrics
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њ

тtotal

уcount
ф	variables
х	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Д

цtotal

чcount
ш
_fn_kwargs
щ	variables
ъ	keras_api"Є
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыnon_trainable_variables
Ѓ	variables
ѓtrainable_variables
ьlayers
∞regularization_losses
 эlayer_regularization_losses
юmetrics
€layer_metrics
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аnon_trainable_variables
і	variables
µtrainable_variables
Бlayers
ґregularization_losses
 Вlayer_regularization_losses
Гmetrics
Дlayer_metrics
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Еnon_trainable_variables
Ї	variables
їtrainable_variables
Жlayers
Љregularization_losses
 Зlayer_regularization_losses
Иmetrics
Йlayer_metrics
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Кnon_trainable_variables
Њ	variables
њtrainable_variables
Лlayers
јregularization_losses
 Мlayer_regularization_losses
Нmetrics
Оlayer_metrics
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Пnon_trainable_variables
¬	variables
√trainable_variables
Рlayers
ƒregularization_losses
 Сlayer_regularization_losses
Тmetrics
Уlayer_metrics
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Фnon_trainable_variables
»	variables
…trainable_variables
Хlayers
 regularization_losses
 Цlayer_regularization_losses
Чmetrics
Шlayer_metrics
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
`0
a1
b2
c3
d4
e5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Щnon_trainable_variables
—	variables
“trainable_variables
Ъlayers
”regularization_losses
 Ыlayer_regularization_losses
Ьmetrics
Эlayer_metrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Юnon_trainable_variables
’	variables
÷trainable_variables
Яlayers
„regularization_losses
 †layer_regularization_losses
°metrics
Ґlayer_metrics
А__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
т0
у1"
trackable_list_wrapper
.
ф	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ц0
ч1"
trackable_list_wrapper
.
щ	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
&:$ 2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
&:$2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
N:L
Ні 2<Adam/token_and_position_embedding_2/embedding_4/embeddings/m
M:K	и 2<Adam/token_and_position_embedding_2/embedding_5/embeddings/m
R:P 
 2>Adam/transformer_block_2/multi_head_attention_2/query/kernel/m
L:J
 2<Adam/transformer_block_2/multi_head_attention_2/query/bias/m
P:N 
 2<Adam/transformer_block_2/multi_head_attention_2/key/kernel/m
J:H
 2:Adam/transformer_block_2/multi_head_attention_2/key/bias/m
R:P 
 2>Adam/transformer_block_2/multi_head_attention_2/value/kernel/m
L:J
 2<Adam/transformer_block_2/multi_head_attention_2/value/bias/m
]:[
  2IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/m
S:Q 2GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/m
%:# d2Adam/dense_8/kernel/m
:d2Adam/dense_8/bias/m
%:#d 2Adam/dense_9/kernel/m
: 2Adam/dense_9/bias/m
B:@ 26Adam/transformer_block_2/layer_normalization_4/gamma/m
A:? 25Adam/transformer_block_2/layer_normalization_4/beta/m
B:@ 26Adam/transformer_block_2/layer_normalization_5/gamma/m
A:? 25Adam/transformer_block_2/layer_normalization_5/beta/m
&:$ 2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
&:$2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
N:L
Ні 2<Adam/token_and_position_embedding_2/embedding_4/embeddings/v
M:K	и 2<Adam/token_and_position_embedding_2/embedding_5/embeddings/v
R:P 
 2>Adam/transformer_block_2/multi_head_attention_2/query/kernel/v
L:J
 2<Adam/transformer_block_2/multi_head_attention_2/query/bias/v
P:N 
 2<Adam/transformer_block_2/multi_head_attention_2/key/kernel/v
J:H
 2:Adam/transformer_block_2/multi_head_attention_2/key/bias/v
R:P 
 2>Adam/transformer_block_2/multi_head_attention_2/value/kernel/v
L:J
 2<Adam/transformer_block_2/multi_head_attention_2/value/bias/v
]:[
  2IAdam/transformer_block_2/multi_head_attention_2/attention_output/kernel/v
S:Q 2GAdam/transformer_block_2/multi_head_attention_2/attention_output/bias/v
%:# d2Adam/dense_8/kernel/v
:d2Adam/dense_8/bias/v
%:#d 2Adam/dense_9/kernel/v
: 2Adam/dense_9/bias/v
B:@ 26Adam/transformer_block_2/layer_normalization_4/gamma/v
A:? 25Adam/transformer_block_2/layer_normalization_4/beta/v
B:@ 26Adam/transformer_block_2/layer_normalization_5/gamma/v
A:? 25Adam/transformer_block_2/layer_normalization_5/beta/v
÷2”
B__inference_model_2_layer_call_and_return_conditional_losses_77773
B__inference_model_2_layer_call_and_return_conditional_losses_77608
B__inference_model_2_layer_call_and_return_conditional_losses_77140
B__inference_model_2_layer_call_and_return_conditional_losses_77085ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
'__inference_model_2_layer_call_fn_77245
'__inference_model_2_layer_call_fn_77822
'__inference_model_2_layer_call_fn_77349
'__inference_model_2_layer_call_fn_77871ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
я2№
 __inference__wrapped_model_76344Ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *'Ґ$
"К
input_3€€€€€€€€€и
ю2ы
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_77895Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
г2а
>__inference_token_and_position_embedding_2_layer_call_fn_77904Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_78052
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_78179∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
†2Э
3__inference_transformer_block_2_layer_call_fn_78216
3__inference_transformer_block_2_layer_call_fn_78253∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
г2а
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_78270
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_78259ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠2™
:__inference_global_average_pooling1d_2_layer_call_fn_78264
:__inference_global_average_pooling1d_2_layer_call_fn_78275ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
E__inference_dropout_10_layer_call_and_return_conditional_losses_78292
E__inference_dropout_10_layer_call_and_return_conditional_losses_78287і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_10_layer_call_fn_78302
*__inference_dropout_10_layer_call_fn_78297і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_dense_10_layer_call_and_return_conditional_losses_78313Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_10_layer_call_fn_78322Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
E__inference_dropout_11_layer_call_and_return_conditional_losses_78334
E__inference_dropout_11_layer_call_and_return_conditional_losses_78339і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_11_layer_call_fn_78344
*__inference_dropout_11_layer_call_fn_78349і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_dense_11_layer_call_and_return_conditional_losses_78360Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_11_layer_call_fn_78369Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 B«
#__inference_signature_wrapper_77408input_3"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
G__inference_sequential_2_layer_call_and_return_conditional_losses_78483
G__inference_sequential_2_layer_call_and_return_conditional_losses_78426
G__inference_sequential_2_layer_call_and_return_conditional_losses_76442
G__inference_sequential_2_layer_call_and_return_conditional_losses_76456ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_2_layer_call_fn_76511
,__inference_sequential_2_layer_call_fn_78496
,__inference_sequential_2_layer_call_fn_76484
,__inference_sequential_2_layer_call_fn_78509ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_8_layer_call_and_return_conditional_losses_78540Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_8_layer_call_fn_78549Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_9_layer_call_and_return_conditional_losses_78579Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_9_layer_call_fn_78588Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 •
 __inference__wrapped_model_76344А=<>?@ABCDEJKFGHILM'(121Ґ.
'Ґ$
"К
input_3€€€€€€€€€и
™ "3™0
.
dense_11"К
dense_11€€€€€€€€€£
C__inference_dense_10_layer_call_and_return_conditional_losses_78313\'(/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_10_layer_call_fn_78322O'(/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€£
C__inference_dense_11_layer_call_and_return_conditional_losses_78360\12/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_11_layer_call_fn_78369O12/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ђ
B__inference_dense_8_layer_call_and_return_conditional_losses_78540fFG4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€и 
™ "*Ґ'
 К
0€€€€€€€€€иd
Ъ Д
'__inference_dense_8_layer_call_fn_78549YFG4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€и 
™ "К€€€€€€€€€иdђ
B__inference_dense_9_layer_call_and_return_conditional_losses_78579fHI4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€иd
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ Д
'__inference_dense_9_layer_call_fn_78588YHI4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€иd
™ "К€€€€€€€€€и •
E__inference_dropout_10_layer_call_and_return_conditional_losses_78287\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ •
E__inference_dropout_10_layer_call_and_return_conditional_losses_78292\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ }
*__inference_dropout_10_layer_call_fn_78297O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "К€€€€€€€€€ }
*__inference_dropout_10_layer_call_fn_78302O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "К€€€€€€€€€ •
E__inference_dropout_11_layer_call_and_return_conditional_losses_78334\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ •
E__inference_dropout_11_layer_call_and_return_conditional_losses_78339\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dropout_11_layer_call_fn_78344O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "К€€€€€€€€€}
*__inference_dropout_11_layer_call_fn_78349O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€‘
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_78259{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ї
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_78270a8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€и 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ђ
:__inference_global_average_pooling1d_2_layer_call_fn_78264nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Т
:__inference_global_average_pooling1d_2_layer_call_fn_78275T8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€и 

 
™ "К€€€€€€€€€ ј
B__inference_model_2_layer_call_and_return_conditional_losses_77085z=<>?@ABCDEJKFGHILM'(129Ґ6
/Ґ,
"К
input_3€€€€€€€€€и
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ј
B__inference_model_2_layer_call_and_return_conditional_losses_77140z=<>?@ABCDEJKFGHILM'(129Ґ6
/Ґ,
"К
input_3€€€€€€€€€и
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ њ
B__inference_model_2_layer_call_and_return_conditional_losses_77608y=<>?@ABCDEJKFGHILM'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€и
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ њ
B__inference_model_2_layer_call_and_return_conditional_losses_77773y=<>?@ABCDEJKFGHILM'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€и
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ш
'__inference_model_2_layer_call_fn_77245m=<>?@ABCDEJKFGHILM'(129Ґ6
/Ґ,
"К
input_3€€€€€€€€€и
p

 
™ "К€€€€€€€€€Ш
'__inference_model_2_layer_call_fn_77349m=<>?@ABCDEJKFGHILM'(129Ґ6
/Ґ,
"К
input_3€€€€€€€€€и
p 

 
™ "К€€€€€€€€€Ч
'__inference_model_2_layer_call_fn_77822l=<>?@ABCDEJKFGHILM'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€и
p

 
™ "К€€€€€€€€€Ч
'__inference_model_2_layer_call_fn_77871l=<>?@ABCDEJKFGHILM'(128Ґ5
.Ґ+
!К
inputs€€€€€€€€€и
p 

 
™ "К€€€€€€€€€¬
G__inference_sequential_2_layer_call_and_return_conditional_losses_76442wFGHICҐ@
9Ґ6
,К)
dense_8_input€€€€€€€€€и 
p

 
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ ¬
G__inference_sequential_2_layer_call_and_return_conditional_losses_76456wFGHICҐ@
9Ґ6
,К)
dense_8_input€€€€€€€€€и 
p 

 
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ ї
G__inference_sequential_2_layer_call_and_return_conditional_losses_78426pFGHI<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€и 
p

 
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ ї
G__inference_sequential_2_layer_call_and_return_conditional_losses_78483pFGHI<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€и 
p 

 
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ Ъ
,__inference_sequential_2_layer_call_fn_76484jFGHICҐ@
9Ґ6
,К)
dense_8_input€€€€€€€€€и 
p

 
™ "К€€€€€€€€€и Ъ
,__inference_sequential_2_layer_call_fn_76511jFGHICҐ@
9Ґ6
,К)
dense_8_input€€€€€€€€€и 
p 

 
™ "К€€€€€€€€€и У
,__inference_sequential_2_layer_call_fn_78496cFGHI<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€и 
p

 
™ "К€€€€€€€€€и У
,__inference_sequential_2_layer_call_fn_78509cFGHI<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€и 
p 

 
™ "К€€€€€€€€€и ≥
#__inference_signature_wrapper_77408Л=<>?@ABCDEJKFGHILM'(12<Ґ9
Ґ 
2™/
-
input_3"К
input_3€€€€€€€€€и"3™0
.
dense_11"К
dense_11€€€€€€€€€Ї
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_77895]=<+Ґ(
!Ґ
К
x€€€€€€€€€и
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ Т
>__inference_token_and_position_embedding_2_layer_call_fn_77904P=<+Ґ(
!Ґ
К
x€€€€€€€€€и
™ "К€€€€€€€€€и  
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_78052x>?@ABCDEJKFGHILM8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€и 
p
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ  
N__inference_transformer_block_2_layer_call_and_return_conditional_losses_78179x>?@ABCDEJKFGHILM8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€и 
p 
™ "*Ґ'
 К
0€€€€€€€€€и 
Ъ Ґ
3__inference_transformer_block_2_layer_call_fn_78216k>?@ABCDEJKFGHILM8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€и 
p
™ "К€€€€€€€€€и Ґ
3__inference_transformer_block_2_layer_call_fn_78253k>?@ABCDEJKFGHILM8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€и 
p 
™ "К€€€€€€€€€и 