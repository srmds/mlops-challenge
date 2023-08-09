# MLOps challenge

## Prerequisites

- [Azure Subscription](https://azure.microsoft.com/)
- [Azure CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public)
- [VSCode](https://code.visualstudio.com/)
- [AZ ML extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-vs-code?view=azureml-api-2)

## Configure

Login with CLI:

```shell
$ az login
```

Export needed arguments:

```shell
GROUP="<RESOURCE_GROUP_NAME>" && LOCATION="<REGION_NAME>" && WORKSPACE="<WORKSPACE_NAME>"
```