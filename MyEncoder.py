import  torch
from torch import nn
class DeltaModulator_old(nn.Module):

    def __init__(self, device:str,delta=0.02,step_mode='s'):
        super().__init__()
        self.delta = delta
        self.step_mode = step_mode
        self.device=device

    def forward(self, x: torch.Tensor,):
        x = x.to(self.device)
        batch_size, num_rows, num_columns = x.shape
        encoded_results = []

        dc = torch.zeros(batch_size, num_rows, device=x.device)

        UP = torch.zeros((batch_size, num_rows, num_columns - 20), dtype=torch.bool, device=x.device)
        DN = torch.zeros_like(UP, dtype=torch.bool)

        for i in range(num_columns - 20):

            upper_bound = dc + self.delta
            lower_bound = dc - self.delta

            UP[:, :, i] = x[:, :, i] > upper_bound
            DN[:, :, i] = x[:, :, i] < lower_bound

            dc = torch.where(UP[:, :, i] | DN[:, :, i], x[:, :, i], dc)

        encoded_output = torch.cat((UP, DN), dim=-1).float()

        additional_columns = x[:, :, 300:320]

        output = torch.cat((encoded_output, additional_columns), dim=-1).float()

        return output
class DeltaModulator(nn.Module):
    def __init__(self, device, delta=0.1, step_mode='s'):
        super().__init__()
        self.initial_delta = delta
        self.step_mode = step_mode
        self.device=device

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        batch_size, num_rows, num_columns = x.shape
        UP = torch.zeros((batch_size, num_rows, num_columns-20), dtype=torch.bool,device=x.device)
        DN = torch.zeros_like(UP, dtype=torch.bool)

        dc = torch.zeros(batch_size, num_rows, device=x.device)

        delta_tensor = torch.full((batch_size, num_rows), self.initial_delta, device=x.device)
        trigger_counter = torch.zeros((batch_size, num_rows), device=x.device)  
        non_trigger_counter = torch.zeros((batch_size, num_rows), device=x.device)  

        for i in range(num_columns-20):
            upper_bound = dc + delta_tensor
            lower_bound = dc - delta_tensor
            
            UP_i = x[:, :, i] > upper_bound
            DN_i = x[:, :, i] < lower_bound
            trigger = UP_i | DN_i

            trigger_counter = torch.where(trigger, trigger_counter + 1, 0)
            non_trigger_counter = torch.where(trigger, 0, non_trigger_counter + 1)
            
            delta_tensor = torch.where(
                trigger_counter >= 3, 
                0.02, 
                delta_tensor
            )
            
            delta_tensor = torch.where(
                non_trigger_counter >= 3,
                0.1,
                delta_tensor
            )
            
            dc = torch.where(trigger, x[:, :, i], dc)
            
            UP[:, :, i] = UP_i
            DN[:, :, i] = DN_i

        encoded_output = torch.cat((UP, DN), dim=-1).float()

        additional_columns = x[:, :, 232:252]

        output = torch.cat((encoded_output, additional_columns), dim=-1).float()

        return output


def data_convert1(df1,device:str):
    """
    对带有 batchsize 维度的输入进行处理：
    - 对0-75列、124-375列和424-599列，每三列合并为一列，合并规则：若三列中有一个1则合并为1，否则为0。
    - 对76-123列和376-423列保留原本的值。
    - 对于第300-319列，单独处理后与处理后的df1拼接。

    参数：
        df1 (torch.Tensor): 输入的三维数据，形状为 (batchsize, rows, columns)

    返回：
        torch.Tensor: 处理后的张量，形状为 (batchsize, rows, 合并后的df1列数 + df2列数)
    """
    batchsize, rows, columns = df1.shape

    df1_part = df1[:, :, :600]  
    df2_part = df1[:, :, 600:620] 

    merged_columns_first_part = (76 // 3) + (76 % 3 != 0) - 1  
    merged_columns_middle_part = (375 - 124 + 1) // 3 + ((375 - 124 + 1) % 3 != 0)  
    merged_columns_last_part = (599 - 424 + 1) // 3 + ((599 - 424 + 1) % 3 != 0) - 1 

    new_columns = merged_columns_first_part + (123 - 76 + 1) + merged_columns_middle_part + (
                423 - 376 + 1) + merged_columns_last_part + 20 
    df_result = torch.zeros((batchsize, rows, new_columns), dtype=torch.int).to(
        torch.device(device))

    first_part = df1_part[:, :, :76]  
    first_part_merged = (first_part[:, :, :75].reshape(batchsize, rows, -1, 3) != 0).any(dim=3).to(torch.int) 
    df_result[:, :, :merged_columns_first_part] = first_part_merged

    df_result[:, :, merged_columns_first_part:merged_columns_first_part + (123 - 76 + 1)] = df1_part[:, :, 76:124]

    middle_part = df1_part[:, :, 124:376]  
    middle_part_merged = (middle_part[:, :, :252].reshape(batchsize, rows, -1, 3) != 0).any(dim=3).to(
        torch.int)  
    df_result[:, :, merged_columns_first_part + (123 - 76 + 1):merged_columns_first_part + (
                123 - 76 + 1) + merged_columns_middle_part] = middle_part_merged

    df_result[:, :,
    merged_columns_first_part + (123 - 76 + 1) + merged_columns_middle_part:merged_columns_first_part + (
                123 - 76 + 1) + merged_columns_middle_part + (423 - 376 + 1)] = df1_part[:, :, 376:424]

    last_part = df1_part[:, :, 424:600]  
    last_part_merged = (last_part[:, :, :174].reshape(batchsize, rows, -1, 3) != 0).any(dim=3).to(torch.int)  
    df_result[:, :, merged_columns_first_part + (123 - 76 + 1) + merged_columns_middle_part + (
                423 - 376 + 1):merged_columns_first_part + (123 - 76 + 1) + merged_columns_middle_part + (
                423 - 376 + 1) + merged_columns_last_part] = last_part_merged

    df_result[:, :, new_columns - 20:] = df2_part[:, :, :20]

    return df_result.float()

def data_convert2(df1,device:str):
    """
    对带有 batchsize 维度的输入进行处理：
    - 对0-49列、100-299列和350-499列，每5列合并为一列，合并规则：若5列中有一个1则合并为1，否则为0。
    - 对50-99列和300-349列保留原本的值。
    - 对于第250-269列，单独处理后与处理后的df1拼接。

    参数：
        df1 (torch.Tensor): 输入的三维数据，形状为 (batchsize, rows, columns)

    返回：
        torch.Tensor: 处理后的张量，形状为 (batchsize, rows, 合并后的df1列数 + df2列数)
    """
    batchsize, rows, columns = df1.shape

    df1_part = df1[:, :, :464]  
    df2_part = df1[:, :, 464:484]  

    merged_columns_first_part = 6
    merged_columns_middle_part = 24
    merged_columns_last_part = 18

    new_columns = merged_columns_first_part + 40 + merged_columns_middle_part + 40 + merged_columns_last_part + 20  
    df_result = torch.zeros((batchsize, rows, new_columns), dtype=torch.int).to(
        torch.device(device))  

    first_part = df1_part[:, :, :48]  
    first_part_merged = (first_part[:, :, :48].reshape(batchsize, rows, -1, 8) != 0).any(dim=3).to(torch.int) 
    df_result[:, :, :merged_columns_first_part] = first_part_merged

    df_result[:, :, merged_columns_first_part:merged_columns_first_part + 40] = df1_part[:, :, 48:88]

    middle_part = df1_part[:, :, 88:280]  
    middle_part_merged = (middle_part[:, :, :192].reshape(batchsize, rows, -1, 8) != 0).any(dim=3).to(
        torch.int)  
    df_result[:, :, merged_columns_first_part + 40:merged_columns_first_part + 40 + merged_columns_middle_part] = middle_part_merged

    df_result[:, :,
    merged_columns_first_part + 40 + merged_columns_middle_part:merged_columns_first_part + 40 + merged_columns_middle_part + 40] = df1_part[:, :, 280:320]

    last_part = df1_part[:, :, 320:464] 
    last_part_merged = (last_part[:, :, :144].reshape(batchsize, rows, -1, 8) != 0).any(dim=3).to(torch.int)  
    df_result[:, :, merged_columns_first_part + 40 + merged_columns_middle_part + 40:merged_columns_first_part + 40 + merged_columns_middle_part + 40 + merged_columns_last_part] = last_part_merged

    df_result[:, :, new_columns - 20:] = df2_part[:, :, :20]
    return df_result.float()