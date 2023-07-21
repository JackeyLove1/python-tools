# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from . import reco_pb2 as reco__pb2


class UserRecommendStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.user_recommend = channel.unary_unary(
        '/UserRecommend/user_recommend',
        request_serializer=reco__pb2.UserRequest.SerializeToString,
        response_deserializer=reco__pb2.ArticleResponse.FromString,
        )


class UserRecommendServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def user_recommend(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_UserRecommendServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'user_recommend': grpc.unary_unary_rpc_method_handler(
          servicer.user_recommend,
          request_deserializer=reco__pb2.UserRequest.FromString,
          response_serializer=reco__pb2.ArticleResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'UserRecommend', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
