import boto3
from botocore.exceptions import EndpointConnectionError

# Inicializando o cliente Textract com a região us-east-1
textract = boto3.client('textract', region_name='us-east-1')

try:
    # Analisando o documento cuidado com o idioma
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': 'aulafiap1', 'Name': 'relatorio_investimentos.pdf'}},
        FeatureTypes=['TABLES', 'FORMS']
    )

    # Processamento da resposta, extraindo o texto do pdf
    def get_text_from_blocks(blocks):
        lines = [block['Text'] for block in blocks if block['BlockType'] == 'LINE']
        return '\n'.join(lines)

    # Extraindo o texto dos blocos
    text = get_text_from_blocks(response['Blocks'])

    # Exibindo o texto extraído
    print(text)

    # Salvando o texto extraído em um arquivo
    with open('texto_extraido.txt', 'w') as file:
        file.write(text)

    print("Análise concluída e texto extraído com sucesso.")

except EndpointConnectionError as e:
    print(f"Erro de conexão com o endpoint: {e}")
    print("Verifique sua conexão à internet, as configurações de rede e se a região especificada está correta.")

except Exception as e:
    print(f"Ocorreu um erro: {e}")