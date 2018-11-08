from lark import Lark
import json

convPath = "param/conv.txt"
maxPoolPath = "param/maxpool.txt"
reluFcPath = "param/relufc.txt"
rnnPath = "param/rnn.txt"
# 目前未考虑的bug：
# 1.return
# 2.do not import tf module
# 3.other api of rnn and fc
# 4.layer num
# 5.batchsize 会有问题
# 6.参数在最后一个会有问题
def extractNetwork(tfFile, debug=False):
    def parseConv(tree, s):
        output = tree.children[0].value
        name = output + '.name'
        outputShape = output + '.shape'
        inputShape = s[tree.children[1].children[0].column - 1:tree.children[2].column - 2] + '.shape'
        kernelShape = s[tree.children[2].children[0].column - 1:tree.children[3].column - 2] + '.shape'
        stride = s[tree.children[3].children[0].column - 1:tree.children[4].column - 2] + '[1:-1]'
        return 'print(%s,"|", %s ,"|", %s ,"|", %s ,"|", %s,file = f)\n' % (
            name, outputShape, inputShape, kernelShape, stride)

    def parseReluFc(tree, s):
        output = tree.children[0].value
        name = output + '.name'
        outputShape = output + '.shape'
        inputShape = s[tree.children[1].children[0].column - 1:tree.children[2].column - 2] + '.shape'
        kernelShape = s[tree.children[2].children[0].column - 1:tree.children[3].column - 2] + '.shape'
        return 'print(%s,"|", %s ,"|", %s ,"|", %s ,file = f)\n' % (name, outputShape, inputShape, kernelShape)

    def parserMaxpool(tree, s):
        output = tree.children[0].value
        name = output + '.name'
        outputShape = output + '.shape'
        inputShape = s[tree.children[1].children[0].column - 1:tree.children[2].column - 2] + '.shape'
        ksize = s[tree.children[2].children[0].column - 1:tree.children[3].column - 2] + '[1:-1]'
        stride = s[tree.children[3].children[0].column - 1:tree.children[4].column - 2] + '[1:-1]'
        return 'print(%s,"|", %s ,"|", %s ,"|", %s ,"|", %s,file = f)\n' % (
            name, outputShape, inputShape, ksize, stride)

    def parserRnn(tree, s):
        outputs = tree.children[0].value
        states = tree.children[1].value
        inputs = s[tree.children[3].children[0].column - 1:tree.children[4].column - 1]  # TODO
        name = outputs + '[0].name'
        output_timeStep = 'len(' + outputs + ')'
        n_output = outputs + '[0].shape'
        input_timeStep = 'len(' + inputs + ')'
        n_input = inputs + '[0].shape'
        n_hidden = states + '[0].shape'
        return 'print(%s,"|",%s,"|", %s ,"|", %s ,"|", %s ,"|", %s ,file = f)\n' % (
            name, input_timeStep, n_input, output_timeStep, n_output, n_hidden)

    l = Lark('''target: conv2d | relufc  | maxpool | rnn
               
                conv2d: VARIABLE "=" "tf.nn.conv2d(" input "," kernel "," strides "," padding [other] ")"
                rnn: VARIABLE "," VARIABLE "=" ["tf."] "rnn.static_rnn(" cell "," input [other] ")"
                relufc: VARIABLE "=" "tf.nn.relu_layer(" input "," kernel "," biases [other] ")"
                maxpool:VARIABLE "=" "tf.nn.max_pool(" input "," ksize "," strides "," padding [other] ")"
                
                other: /,[^)]+/
                input: exp
                cell: ["cell" "="] exp
                kernel: ["kernel" "="] exp
                ksize: ["ksize" "="]exp
                strides: ["strides" "="] exp
                padding: ["padding" "="] exp 
                biases: ["biases" "="] exp           
                               
                exp:  expy
                    |  expf
                    | normexp                    
                expy: "(" exps ")"
                expf:"[" exps "]"
                VARIABLE:   /[a-zA-Z_][a-zA-Z0-9_]*/
                normexp: /[^,()[\]]+/
                exps: ( expy | sexp | expf ) +
                sexp:/[^()[\]]+/
                %ignore " "           // Disregard spaces in text

             ''', start='target', propagate_positions=True)
    l.parse("activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)")
    newFile = ''
    open("param/conv.txt", "w")
    open("param/relufc.txt", "w")
    open("param/maxpool.txt", "w")
    open("param/rnn.txt", "w")
    with open(tfFile, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            i = 0
            while i != len(line):
                if line[i] != ' ' and line[i] != '\t':
                    break
                i += 1
            tab = line[0:i]
            sentence = line[i:].strip('\n').replace(' ', '')
            tree = ''
            try:
                tree = l.parse(sentence)
            except Exception:
                pass
            newFile += line
            if tree != '':
                if tree.children[0].data == 'conv2d':
                    newFile += tab + 'f = open("' + convPath + '", "a")\n'
                    newFile += tab + parseConv(tree.children[0], sentence)
                elif tree.children[0].data == 'relufc':
                    newFile += tab + 'f = open("' + reluFcPath + '", "a")\n'
                    newFile += tab + parseReluFc(tree.children[0], sentence)
                elif tree.children[0].data == 'maxpool':
                    newFile += tab + 'f = open("' + maxPoolPath + '", "a")\n'
                    newFile += tab + parserMaxpool(tree.children[0], sentence)
                elif tree.children[0].data == 'rnn':
                    newFile += tab + 'f = open("' + rnnPath + '", "a")\n'
                    newFile += tab + parserRnn(tree.children[0], sentence)

    with open('newFile.py', 'w', encoding="utf-8") as f:
        f.write(newFile)
    import os
    os.system('C:\\Users\\user\\Anaconda3\\envs\\tf\\python.exe newFile.py')
    if not debug:
        os.remove("newFile.py")


def modify_format(modelName):
    def transform(d):
        return str(json.dumps(d, indent=2)).replace(',', '').replace('"', '').replace(':', ' =')

    def modify_conv():
        with open(convPath, 'r') as f:
            for line in f.readlines():
                name, outputShape, inputShape, kernelShape, stride = line.split('|')
                outputShape = outputShape[2:-2].split(',')
                inputShape = inputShape[2:-2].split(',')
                kernelShape = kernelShape[2:-2].split(',')
                stride = stride[2:-2].split(',')

                Input_param, Output_param, Weight_param = dict(), dict(), dict()
                Input_param['input_batch'], Input_param['input_x'], Input_param['input_y'], Input_param['input_channel'] \
                    = inputShape
                Output_param['output_batch'], Output_param['output_x'], Output_param['output_y'], Output_param[
                    'output_channel'] \
                    = outputShape
                Weight_param['weight_x'], Weight_param['weight_y'], Weight_param['weight_channel'], Weight_param[
                    'weight_number'] \
                    = kernelShape
                Weight_param['stride_x'], Weight_param['stride_y'] = stride
                filename = 'param/_' + name.replace('/', '').replace(' ', '')[0:-2] + '.txt'
                with open(filename, 'w') as f2:
                    f2.write('Model_Name = ' + modelName + '\n')
                    f2.write('Layer_Type = CONV\n')
                    f2.write('Layer_Number = ' + name.split('/')[0])
                    f2.write('\n\nInput_parameter')
                    f2.write(transform(Input_param))
                    f2.write('\n\nOutput_parameter')
                    f2.write(transform(Output_param))
                    f2.write('\n\nWeight_parameter')
                    f2.write(transform(Weight_param))

    def modify_maxpool():
        with open(maxPoolPath, 'r') as f:
            for line in f.readlines():
                name, outputShape, inputShape, kernelShape, stride = line.split('|')
                outputShape = outputShape[2:-2].split(',')
                inputShape = inputShape[2:-2].split(',')
                kernelShape = kernelShape[2:-2].split(',')
                stride = stride[2:-2].split(',')

                Input_param, Output_param, Weight_param = dict(), dict(), dict()
                Input_param['input_batch'], Input_param['input_x'], Input_param['input_y'], Input_param['input_channel'] \
                    = inputShape
                Output_param['output_batch'], Output_param['output_x'], Output_param['output_y'], Output_param[
                    'output_channel'] \
                    = outputShape
                Weight_param['ksize_x'], Weight_param['ksize_y'] = kernelShape
                Weight_param['stride_x'], Weight_param['stride_y'] = stride
                filename = 'param/_' + name.replace('/', '').replace(' ', '')[0:-2] + '.txt'
                with open(filename, 'w') as f2:
                    f2.write('Model_Name = ' + modelName + '\n')
                    f2.write('Layer_Type = MAXPOOL\n')
                    f2.write('Layer_Number = ' + name.split('/')[0])
                    f2.write('\n\nInput_parameter')
                    f2.write(transform(Input_param))
                    f2.write('\n\nOutput_parameter')
                    f2.write(transform(Output_param))
                    f2.write('\n\nWeight_parameter')
                    f2.write(transform(Weight_param))

    def modify_relufc():
        with open(reluFcPath, 'r') as f:
            for line in f.readlines():
                name, outputShape, inputShape, kernelShape = line.split('|')
                outputShape = outputShape[2:-2].split(',')
                inputShape = inputShape[2:-2].split(',')
                kernelShape = kernelShape[2:-2].split(',')

                Input_param, Output_param, Weight_param = dict(), dict(), dict()
                Input_param['input_batch'], Input_param['input_length'] = inputShape
                Output_param['output_batch'], Output_param['output_length'] = outputShape
                Weight_param['dimension_1'], Weight_param['dimension_2'] = kernelShape
                filename = 'param/_' + name.replace('/', '').replace(' ', '')[0:-2] + '.txt'
                with open(filename, 'w') as f2:
                    f2.write('Model_Name = ' + modelName + '\n')
                    f2.write('Layer_Type = RELEFC\n')
                    f2.write('Layer_Number = ' + name.split('/')[0])
                    f2.write('\n\nInput_parameter')
                    f2.write(transform(Input_param))
                    f2.write('\n\nOutput_parameter')
                    f2.write(transform(Output_param))
                    f2.write('\n\nWeight_parameter')
                    f2.write(transform(Weight_param))

    def modify_rnn():
        with open(rnnPath, 'r') as f:
            for line in f.readlines():
                name, input_Step, n_input, output_Step, n_output, n_hidden = line.split('|')
                n_input = n_input[2:-2].split(',')
                n_output = n_output[2:-2].split(',')
                n_hidden = n_hidden[2:-2].split(',')

                Input_param, Output_param, Hidden_param = dict(), dict(), dict()
                Input_param['input_batch'], Input_param['input_step'], Input_param['input_dim'] = n_input[0], input_Step,n_input[1]
                Output_param['output_batch'], Output_param['output_step'],Output_param['output_dim']  = n_output[0], output_Step,n_output[1]
                Hidden_param['hidden_batch'], Hidden_param['hidden_dim'] = n_hidden
                filename = 'param/_' + name.replace('/', '').replace(' ', '')[0:-2] + '.txt'
                with open(filename, 'w') as f2:
                    f2.write('Model_Name = ' + modelName + '\n')
                    f2.write('Layer_Type = rnn_unit\n')
                    f2.write('Layer_Number = ' + name.split('/')[-1])
                    f2.write('\n\nInput_parameter')
                    f2.write(transform(Input_param))
                    f2.write('\n\nOutput_parameter')
                    f2.write(transform(Output_param))
                    f2.write('\n\nHidden_parameter')
                    f2.write(transform(Hidden_param))

    modify_conv()
    modify_maxpool()
    modify_relufc()
    modify_rnn()

extractNetwork("vgg.py",True)
modify_format("vgg")
