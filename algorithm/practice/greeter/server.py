# python3  -m grpc_tools.protoc -I../protos --python_out=. --pyi_out=. --grpc_python_out=. ../protos/hello.proto
from concurrent import futures
import logging
import grpc
import hello_pb2
import hello_pb2_grpc

class Greeter(hello_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return hello_pb2.HelloReply(f"Hello,{request.name}!")

