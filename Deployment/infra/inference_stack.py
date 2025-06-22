import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_logs as logs,
)

class InferenceStack(Stack):
    def __init__(self, scope):
        super().__init__(scope, 'InferenceStack')

        inference_lambda = _lambda.DockerImageFunction(
            self, 'InferenceLambda',
            function_name='Inference_AAI510',
            description='Deployment of the TabTransformer model for AAI510 final project',
            code=_lambda.DockerImageCode.from_image_asset(
                directory='lambda',
            ),
            memory_size=1024,
            timeout=Duration.seconds(30),
            environment={
                'API_KEY': '1234Example'
            },
            log_retention=logs.RetentionDays.ONE_WEEK
        )

        function_url = inference_lambda.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE,
            cors=_lambda.FunctionUrlCorsOptions(
                allowed_origins=['*'],
                allowed_methods=[_lambda.HttpMethod.GET, _lambda.HttpMethod.POST],
                allowed_headers=['*']
            )
        )

        cdk.CfnOutput(self, 'LambdaFunctionUrl', value=function_url.url)