// ==========================================================================
// RISC-V CPU Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: deepseek
// Generated: 2026-11-04 17:30:41
// Oracle: bdd
// Program: 865 instructions, 66 checks
// ==========================================================================

`timescale 1ns/1ps

module riscv_cpu_tb;
    reg clk; reg rst_n;
    wire [31:0] imem_addr; reg [31:0] imem_data;
    wire [31:0] dmem_addr, dmem_wdata; wire dmem_wen; wire [3:0] dmem_be;
    reg  [31:0] dmem_rdata;
    wire [31:0] debug_pc, debug_inst;
    integer total, passed, failed, i;
    reg [31:0] imem [0:1023];
    reg [31:0] dmem [0:1023];

    riscv_cpu dut (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .dmem_addr(dmem_addr), .dmem_wdata(dmem_wdata),
        .dmem_wen(dmem_wen), .dmem_be(dmem_be), .dmem_rdata(dmem_rdata),
        .debug_pc(debug_pc), .debug_inst(debug_inst)
    );

    initial clk = 0;
    always #5 clk = ~clk;
    always @(*) imem_data = imem[imem_addr[11:2]];
    always @(posedge clk) begin
        if (dmem_wen) dmem[dmem_addr[11:2]] <= dmem_wdata;
        dmem_rdata <= dmem[dmem_addr[11:2]];
    end

    initial begin
        total = 0; passed = 0; failed = 0; rst_n = 0;
        for (i = 0; i < 1024; i = i + 1) imem[i] = 32'h00000013;
        for (i = 0; i < 1024; i = i + 1) dmem[i] = 32'h0;
        dmem[64] = 32'h00000000;
        dmem[65] = 32'h00000001;
        dmem[66] = 32'hFFFFFFFF;
        dmem[67] = 32'h000007FF;
        dmem[68] = 32'hFFFFF800;
        dmem[128] = 32'h000004D2;
        imem[0] = 32'h00A00093;
        imem[1] = 32'h01400113;
        imem[2] = 32'h40000F13;
        imem[3] = 32'h00000013;
        imem[4] = 32'h00000013;
        imem[5] = 32'h00000013;
        imem[6] = 32'h002081B3;
        imem[7] = 32'h00000013;
        imem[8] = 32'h00000013;
        imem[9] = 32'h00000013;
        imem[10] = 32'h003F2023;
        imem[11] = 32'h00000013;
        imem[12] = 32'h00000013;
        imem[13] = 32'h00000013;
        imem[14] = 32'h00000093;
        imem[15] = 32'h00000113;
        imem[16] = 32'h40400F13;
        imem[17] = 32'h00000013;
        imem[18] = 32'h00000013;
        imem[19] = 32'h00000013;
        imem[20] = 32'h002081B3;
        imem[21] = 32'h00000013;
        imem[22] = 32'h00000013;
        imem[23] = 32'h00000013;
        imem[24] = 32'h003F2023;
        imem[25] = 32'h00000013;
        imem[26] = 32'h00000013;
        imem[27] = 32'h00000013;
        imem[28] = 32'h7FF00093;
        imem[29] = 32'h00100113;
        imem[30] = 32'h40800F13;
        imem[31] = 32'h00000013;
        imem[32] = 32'h00000013;
        imem[33] = 32'h00000013;
        imem[34] = 32'h002081B3;
        imem[35] = 32'h00000013;
        imem[36] = 32'h00000013;
        imem[37] = 32'h00000013;
        imem[38] = 32'h003F2023;
        imem[39] = 32'h00000013;
        imem[40] = 32'h00000013;
        imem[41] = 32'h00000013;
        imem[42] = 32'h80000093;
        imem[43] = 32'hFFF00113;
        imem[44] = 32'h40C00F13;
        imem[45] = 32'h00000013;
        imem[46] = 32'h00000013;
        imem[47] = 32'h00000013;
        imem[48] = 32'h002081B3;
        imem[49] = 32'h00000013;
        imem[50] = 32'h00000013;
        imem[51] = 32'h00000013;
        imem[52] = 32'h003F2023;
        imem[53] = 32'h00000013;
        imem[54] = 32'h00000013;
        imem[55] = 32'h00000013;
        imem[56] = 32'h01E00093;
        imem[57] = 32'h00A00113;
        imem[58] = 32'h41000F13;
        imem[59] = 32'h00000013;
        imem[60] = 32'h00000013;
        imem[61] = 32'h00000013;
        imem[62] = 32'h402081B3;
        imem[63] = 32'h00000013;
        imem[64] = 32'h00000013;
        imem[65] = 32'h00000013;
        imem[66] = 32'h003F2023;
        imem[67] = 32'h00000013;
        imem[68] = 32'h00000013;
        imem[69] = 32'h00000013;
        imem[70] = 32'hFFF00093;
        imem[71] = 32'h00100113;
        imem[72] = 32'h41400F13;
        imem[73] = 32'h00000013;
        imem[74] = 32'h00000013;
        imem[75] = 32'h00000013;
        imem[76] = 32'h402081B3;
        imem[77] = 32'h00000013;
        imem[78] = 32'h00000013;
        imem[79] = 32'h00000013;
        imem[80] = 32'h003F2023;
        imem[81] = 32'h00000013;
        imem[82] = 32'h00000013;
        imem[83] = 32'h00000013;
        imem[84] = 32'h7FF00093;
        imem[85] = 32'h80000113;
        imem[86] = 32'h41800F13;
        imem[87] = 32'h00000013;
        imem[88] = 32'h00000013;
        imem[89] = 32'h00000013;
        imem[90] = 32'h402081B3;
        imem[91] = 32'h00000013;
        imem[92] = 32'h00000013;
        imem[93] = 32'h00000013;
        imem[94] = 32'h003F2023;
        imem[95] = 32'h00000013;
        imem[96] = 32'h00000013;
        imem[97] = 32'h00000013;
        imem[98] = 32'h00000093;
        imem[99] = 32'h00100113;
        imem[100] = 32'h41C00F13;
        imem[101] = 32'h00000013;
        imem[102] = 32'h00000013;
        imem[103] = 32'h00000013;
        imem[104] = 32'h402081B3;
        imem[105] = 32'h00000013;
        imem[106] = 32'h00000013;
        imem[107] = 32'h00000013;
        imem[108] = 32'h003F2023;
        imem[109] = 32'h00000013;
        imem[110] = 32'h00000013;
        imem[111] = 32'h00000013;
        imem[112] = 32'h0FF00093;
        imem[113] = 32'h00F00113;
        imem[114] = 32'h42000F13;
        imem[115] = 32'h00000013;
        imem[116] = 32'h00000013;
        imem[117] = 32'h00000013;
        imem[118] = 32'h0020F1B3;
        imem[119] = 32'h00000013;
        imem[120] = 32'h00000013;
        imem[121] = 32'h00000013;
        imem[122] = 32'h003F2023;
        imem[123] = 32'h00000013;
        imem[124] = 32'h00000013;
        imem[125] = 32'h00000013;
        imem[126] = 32'hFFF00093;
        imem[127] = 32'h7FF00113;
        imem[128] = 32'h42400F13;
        imem[129] = 32'h00000013;
        imem[130] = 32'h00000013;
        imem[131] = 32'h00000013;
        imem[132] = 32'h0020F1B3;
        imem[133] = 32'h00000013;
        imem[134] = 32'h00000013;
        imem[135] = 32'h00000013;
        imem[136] = 32'h003F2023;
        imem[137] = 32'h00000013;
        imem[138] = 32'h00000013;
        imem[139] = 32'h00000013;
        imem[140] = 32'h00000093;
        imem[141] = 32'hFFF00113;
        imem[142] = 32'h42800F13;
        imem[143] = 32'h00000013;
        imem[144] = 32'h00000013;
        imem[145] = 32'h00000013;
        imem[146] = 32'h0020F1B3;
        imem[147] = 32'h00000013;
        imem[148] = 32'h00000013;
        imem[149] = 32'h00000013;
        imem[150] = 32'h003F2023;
        imem[151] = 32'h00000013;
        imem[152] = 32'h00000013;
        imem[153] = 32'h00000013;
        imem[154] = 32'h0F000093;
        imem[155] = 32'h00F00113;
        imem[156] = 32'h42C00F13;
        imem[157] = 32'h00000013;
        imem[158] = 32'h00000013;
        imem[159] = 32'h00000013;
        imem[160] = 32'h0020E1B3;
        imem[161] = 32'h00000013;
        imem[162] = 32'h00000013;
        imem[163] = 32'h00000013;
        imem[164] = 32'h003F2023;
        imem[165] = 32'h00000013;
        imem[166] = 32'h00000013;
        imem[167] = 32'h00000013;
        imem[168] = 32'h00000093;
        imem[169] = 32'h7FF00113;
        imem[170] = 32'h43000F13;
        imem[171] = 32'h00000013;
        imem[172] = 32'h00000013;
        imem[173] = 32'h00000013;
        imem[174] = 32'h0020E1B3;
        imem[175] = 32'h00000013;
        imem[176] = 32'h00000013;
        imem[177] = 32'h00000013;
        imem[178] = 32'h003F2023;
        imem[179] = 32'h00000013;
        imem[180] = 32'h00000013;
        imem[181] = 32'h00000013;
        imem[182] = 32'h80000093;
        imem[183] = 32'h00000113;
        imem[184] = 32'h43400F13;
        imem[185] = 32'h00000013;
        imem[186] = 32'h00000013;
        imem[187] = 32'h00000013;
        imem[188] = 32'h0020E1B3;
        imem[189] = 32'h00000013;
        imem[190] = 32'h00000013;
        imem[191] = 32'h00000013;
        imem[192] = 32'h003F2023;
        imem[193] = 32'h00000013;
        imem[194] = 32'h00000013;
        imem[195] = 32'h00000013;
        imem[196] = 32'hFFF00093;
        imem[197] = 32'h7FF00113;
        imem[198] = 32'h43800F13;
        imem[199] = 32'h00000013;
        imem[200] = 32'h00000013;
        imem[201] = 32'h00000013;
        imem[202] = 32'h0020C1B3;
        imem[203] = 32'h00000013;
        imem[204] = 32'h00000013;
        imem[205] = 32'h00000013;
        imem[206] = 32'h003F2023;
        imem[207] = 32'h00000013;
        imem[208] = 32'h00000013;
        imem[209] = 32'h00000013;
        imem[210] = 32'h00100093;
        imem[211] = 32'hFFF00113;
        imem[212] = 32'h43C00F13;
        imem[213] = 32'h00000013;
        imem[214] = 32'h00000013;
        imem[215] = 32'h00000013;
        imem[216] = 32'h0020C1B3;
        imem[217] = 32'h00000013;
        imem[218] = 32'h00000013;
        imem[219] = 32'h00000013;
        imem[220] = 32'h003F2023;
        imem[221] = 32'h00000013;
        imem[222] = 32'h00000013;
        imem[223] = 32'h00000013;
        imem[224] = 32'h80000093;
        imem[225] = 32'h7FF00113;
        imem[226] = 32'h44000F13;
        imem[227] = 32'h00000013;
        imem[228] = 32'h00000013;
        imem[229] = 32'h00000013;
        imem[230] = 32'h0020C1B3;
        imem[231] = 32'h00000013;
        imem[232] = 32'h00000013;
        imem[233] = 32'h00000013;
        imem[234] = 32'h003F2023;
        imem[235] = 32'h00000013;
        imem[236] = 32'h00000013;
        imem[237] = 32'h00000013;
        imem[238] = 32'hFFF00093;
        imem[239] = 32'h00100113;
        imem[240] = 32'h44400F13;
        imem[241] = 32'h00000013;
        imem[242] = 32'h00000013;
        imem[243] = 32'h00000013;
        imem[244] = 32'h0020A1B3;
        imem[245] = 32'h00000013;
        imem[246] = 32'h00000013;
        imem[247] = 32'h00000013;
        imem[248] = 32'h003F2023;
        imem[249] = 32'h00000013;
        imem[250] = 32'h00000013;
        imem[251] = 32'h00000013;
        imem[252] = 32'h00500093;
        imem[253] = 32'h00300113;
        imem[254] = 32'h44800F13;
        imem[255] = 32'h00000013;
        imem[256] = 32'h00000013;
        imem[257] = 32'h00000013;
        imem[258] = 32'h0020A1B3;
        imem[259] = 32'h00000013;
        imem[260] = 32'h00000013;
        imem[261] = 32'h00000013;
        imem[262] = 32'h003F2023;
        imem[263] = 32'h00000013;
        imem[264] = 32'h00000013;
        imem[265] = 32'h00000013;
        imem[266] = 32'h80000093;
        imem[267] = 32'h7FF00113;
        imem[268] = 32'h44C00F13;
        imem[269] = 32'h00000013;
        imem[270] = 32'h00000013;
        imem[271] = 32'h00000013;
        imem[272] = 32'h0020A1B3;
        imem[273] = 32'h00000013;
        imem[274] = 32'h00000013;
        imem[275] = 32'h00000013;
        imem[276] = 32'h003F2023;
        imem[277] = 32'h00000013;
        imem[278] = 32'h00000013;
        imem[279] = 32'h00000013;
        imem[280] = 32'h7FF00093;
        imem[281] = 32'h80000113;
        imem[282] = 32'h45000F13;
        imem[283] = 32'h00000013;
        imem[284] = 32'h00000013;
        imem[285] = 32'h00000013;
        imem[286] = 32'h0020A1B3;
        imem[287] = 32'h00000013;
        imem[288] = 32'h00000013;
        imem[289] = 32'h00000013;
        imem[290] = 32'h003F2023;
        imem[291] = 32'h00000013;
        imem[292] = 32'h00000013;
        imem[293] = 32'h00000013;
        imem[294] = 32'h00000093;
        imem[295] = 32'h00000113;
        imem[296] = 32'h45400F13;
        imem[297] = 32'h00000013;
        imem[298] = 32'h00000013;
        imem[299] = 32'h00000013;
        imem[300] = 32'h0020A1B3;
        imem[301] = 32'h00000013;
        imem[302] = 32'h00000013;
        imem[303] = 32'h00000013;
        imem[304] = 32'h003F2023;
        imem[305] = 32'h00000013;
        imem[306] = 32'h00000013;
        imem[307] = 32'h00000013;
        imem[308] = 32'hFFF00093;
        imem[309] = 32'h80000113;
        imem[310] = 32'h45800F13;
        imem[311] = 32'h00000013;
        imem[312] = 32'h00000013;
        imem[313] = 32'h00000013;
        imem[314] = 32'h0020A1B3;
        imem[315] = 32'h00000013;
        imem[316] = 32'h00000013;
        imem[317] = 32'h00000013;
        imem[318] = 32'h003F2023;
        imem[319] = 32'h00000013;
        imem[320] = 32'h00000013;
        imem[321] = 32'h00000013;
        imem[322] = 32'hFFF00093;
        imem[323] = 32'h45C00F13;
        imem[324] = 32'h00000013;
        imem[325] = 32'h00000013;
        imem[326] = 32'h00000013;
        imem[327] = 32'h0000A193;
        imem[328] = 32'h00000013;
        imem[329] = 32'h00000013;
        imem[330] = 32'h00000013;
        imem[331] = 32'h003F2023;
        imem[332] = 32'h00000013;
        imem[333] = 32'h00000013;
        imem[334] = 32'h00000013;
        imem[335] = 32'h80000093;
        imem[336] = 32'h46000F13;
        imem[337] = 32'h00000013;
        imem[338] = 32'h00000013;
        imem[339] = 32'h00000013;
        imem[340] = 32'hFFF0A193;
        imem[341] = 32'h00000013;
        imem[342] = 32'h00000013;
        imem[343] = 32'h00000013;
        imem[344] = 32'h003F2023;
        imem[345] = 32'h00000013;
        imem[346] = 32'h00000013;
        imem[347] = 32'h00000013;
        imem[348] = 32'h00000093;
        imem[349] = 32'h46400F13;
        imem[350] = 32'h00000013;
        imem[351] = 32'h00000013;
        imem[352] = 32'h00000013;
        imem[353] = 32'h0010A193;
        imem[354] = 32'h00000013;
        imem[355] = 32'h00000013;
        imem[356] = 32'h00000013;
        imem[357] = 32'h003F2023;
        imem[358] = 32'h00000013;
        imem[359] = 32'h00000013;
        imem[360] = 32'h00000013;
        imem[361] = 32'h7FF00093;
        imem[362] = 32'h46800F13;
        imem[363] = 32'h00000013;
        imem[364] = 32'h00000013;
        imem[365] = 32'h00000013;
        imem[366] = 32'h7FF0A193;
        imem[367] = 32'h00000013;
        imem[368] = 32'h00000013;
        imem[369] = 32'h00000013;
        imem[370] = 32'h003F2023;
        imem[371] = 32'h00000013;
        imem[372] = 32'h00000013;
        imem[373] = 32'h00000013;
        imem[374] = 32'h80000093;
        imem[375] = 32'h46C00F13;
        imem[376] = 32'h00000013;
        imem[377] = 32'h00000013;
        imem[378] = 32'h00000013;
        imem[379] = 32'h8000A193;
        imem[380] = 32'h00000013;
        imem[381] = 32'h00000013;
        imem[382] = 32'h00000013;
        imem[383] = 32'h003F2023;
        imem[384] = 32'h00000013;
        imem[385] = 32'h00000013;
        imem[386] = 32'h00000013;
        imem[387] = 32'h03200093;
        imem[388] = 32'h47000F13;
        imem[389] = 32'h00000013;
        imem[390] = 32'h00000013;
        imem[391] = 32'h00000013;
        imem[392] = 32'h06408193;
        imem[393] = 32'h00000013;
        imem[394] = 32'h00000013;
        imem[395] = 32'h00000013;
        imem[396] = 32'h003F2023;
        imem[397] = 32'h00000013;
        imem[398] = 32'h00000013;
        imem[399] = 32'h00000013;
        imem[400] = 32'h00000093;
        imem[401] = 32'h47400F13;
        imem[402] = 32'h00000013;
        imem[403] = 32'h00000013;
        imem[404] = 32'h00000013;
        imem[405] = 32'hFFF08193;
        imem[406] = 32'h00000013;
        imem[407] = 32'h00000013;
        imem[408] = 32'h00000013;
        imem[409] = 32'h003F2023;
        imem[410] = 32'h00000013;
        imem[411] = 32'h00000013;
        imem[412] = 32'h00000013;
        imem[413] = 32'h00100093;
        imem[414] = 32'h47800F13;
        imem[415] = 32'h00000013;
        imem[416] = 32'h00000013;
        imem[417] = 32'h00000013;
        imem[418] = 32'h7FF08193;
        imem[419] = 32'h00000013;
        imem[420] = 32'h00000013;
        imem[421] = 32'h00000013;
        imem[422] = 32'h003F2023;
        imem[423] = 32'h00000013;
        imem[424] = 32'h00000013;
        imem[425] = 32'h00000013;
        imem[426] = 32'h80000093;
        imem[427] = 32'h47C00F13;
        imem[428] = 32'h00000013;
        imem[429] = 32'h00000013;
        imem[430] = 32'h00000013;
        imem[431] = 32'h80008193;
        imem[432] = 32'h00000013;
        imem[433] = 32'h00000013;
        imem[434] = 32'h00000013;
        imem[435] = 32'h003F2023;
        imem[436] = 32'h00000013;
        imem[437] = 32'h00000013;
        imem[438] = 32'h00000013;
        imem[439] = 32'h0FF00093;
        imem[440] = 32'h48000F13;
        imem[441] = 32'h00000013;
        imem[442] = 32'h00000013;
        imem[443] = 32'h00000013;
        imem[444] = 32'h00F0F193;
        imem[445] = 32'h00000013;
        imem[446] = 32'h00000013;
        imem[447] = 32'h00000013;
        imem[448] = 32'h003F2023;
        imem[449] = 32'h00000013;
        imem[450] = 32'h00000013;
        imem[451] = 32'h00000013;
        imem[452] = 32'h7FF00093;
        imem[453] = 32'h48400F13;
        imem[454] = 32'h00000013;
        imem[455] = 32'h00000013;
        imem[456] = 32'h00000013;
        imem[457] = 32'hFFF0F193;
        imem[458] = 32'h00000013;
        imem[459] = 32'h00000013;
        imem[460] = 32'h00000013;
        imem[461] = 32'h003F2023;
        imem[462] = 32'h00000013;
        imem[463] = 32'h00000013;
        imem[464] = 32'h00000013;
        imem[465] = 32'h0F000093;
        imem[466] = 32'h48800F13;
        imem[467] = 32'h00000013;
        imem[468] = 32'h00000013;
        imem[469] = 32'h00000013;
        imem[470] = 32'h00F0E193;
        imem[471] = 32'h00000013;
        imem[472] = 32'h00000013;
        imem[473] = 32'h00000013;
        imem[474] = 32'h003F2023;
        imem[475] = 32'h00000013;
        imem[476] = 32'h00000013;
        imem[477] = 32'h00000013;
        imem[478] = 32'h00000093;
        imem[479] = 32'h48C00F13;
        imem[480] = 32'h00000013;
        imem[481] = 32'h00000013;
        imem[482] = 32'h00000013;
        imem[483] = 32'h7FF0E193;
        imem[484] = 32'h00000013;
        imem[485] = 32'h00000013;
        imem[486] = 32'h00000013;
        imem[487] = 32'h003F2023;
        imem[488] = 32'h00000013;
        imem[489] = 32'h00000013;
        imem[490] = 32'h00000013;
        imem[491] = 32'hFFF00093;
        imem[492] = 32'h49000F13;
        imem[493] = 32'h00000013;
        imem[494] = 32'h00000013;
        imem[495] = 32'h00000013;
        imem[496] = 32'h8000E193;
        imem[497] = 32'h00000013;
        imem[498] = 32'h00000013;
        imem[499] = 32'h00000013;
        imem[500] = 32'h003F2023;
        imem[501] = 32'h00000013;
        imem[502] = 32'h00000013;
        imem[503] = 32'h00000013;
        imem[504] = 32'h0FF00093;
        imem[505] = 32'h49400F13;
        imem[506] = 32'h00000013;
        imem[507] = 32'h00000013;
        imem[508] = 32'h00000013;
        imem[509] = 32'h00F0C193;
        imem[510] = 32'h00000013;
        imem[511] = 32'h00000013;
        imem[512] = 32'h00000013;
        imem[513] = 32'h003F2023;
        imem[514] = 32'h00000013;
        imem[515] = 32'h00000013;
        imem[516] = 32'h00000013;
        imem[517] = 32'h7FF00093;
        imem[518] = 32'h49800F13;
        imem[519] = 32'h00000013;
        imem[520] = 32'h00000013;
        imem[521] = 32'h00000013;
        imem[522] = 32'hFFF0C193;
        imem[523] = 32'h00000013;
        imem[524] = 32'h00000013;
        imem[525] = 32'h00000013;
        imem[526] = 32'h003F2023;
        imem[527] = 32'h00000013;
        imem[528] = 32'h00000013;
        imem[529] = 32'h00000013;
        imem[530] = 32'hFFF00093;
        imem[531] = 32'h49C00F13;
        imem[532] = 32'h00000013;
        imem[533] = 32'h00000013;
        imem[534] = 32'h00000013;
        imem[535] = 32'h8000C193;
        imem[536] = 32'h00000013;
        imem[537] = 32'h00000013;
        imem[538] = 32'h00000013;
        imem[539] = 32'h003F2023;
        imem[540] = 32'h00000013;
        imem[541] = 32'h00000013;
        imem[542] = 32'h00000013;
        imem[543] = 32'h10000093;
        imem[544] = 32'h4A000F13;
        imem[545] = 32'h00000013;
        imem[546] = 32'h00000013;
        imem[547] = 32'h00000013;
        imem[548] = 32'h0000A183;
        imem[549] = 32'h00000013;
        imem[550] = 32'h00000013;
        imem[551] = 32'h00000013;
        imem[552] = 32'h003F2023;
        imem[553] = 32'h00000013;
        imem[554] = 32'h00000013;
        imem[555] = 32'h00000013;
        imem[556] = 32'h10400093;
        imem[557] = 32'h4A400F13;
        imem[558] = 32'h00000013;
        imem[559] = 32'h00000013;
        imem[560] = 32'h00000013;
        imem[561] = 32'h0000A183;
        imem[562] = 32'h00000013;
        imem[563] = 32'h00000013;
        imem[564] = 32'h00000013;
        imem[565] = 32'h003F2023;
        imem[566] = 32'h00000013;
        imem[567] = 32'h00000013;
        imem[568] = 32'h00000013;
        imem[569] = 32'h10800093;
        imem[570] = 32'h4A800F13;
        imem[571] = 32'h00000013;
        imem[572] = 32'h00000013;
        imem[573] = 32'h00000013;
        imem[574] = 32'h0000A183;
        imem[575] = 32'h00000013;
        imem[576] = 32'h00000013;
        imem[577] = 32'h00000013;
        imem[578] = 32'h003F2023;
        imem[579] = 32'h00000013;
        imem[580] = 32'h00000013;
        imem[581] = 32'h00000013;
        imem[582] = 32'h10C00093;
        imem[583] = 32'h4AC00F13;
        imem[584] = 32'h00000013;
        imem[585] = 32'h00000013;
        imem[586] = 32'h00000013;
        imem[587] = 32'h0000A183;
        imem[588] = 32'h00000013;
        imem[589] = 32'h00000013;
        imem[590] = 32'h00000013;
        imem[591] = 32'h003F2023;
        imem[592] = 32'h00000013;
        imem[593] = 32'h00000013;
        imem[594] = 32'h00000013;
        imem[595] = 32'h11000093;
        imem[596] = 32'h4B000F13;
        imem[597] = 32'h00000013;
        imem[598] = 32'h00000013;
        imem[599] = 32'h00000013;
        imem[600] = 32'h0000A183;
        imem[601] = 32'h00000013;
        imem[602] = 32'h00000013;
        imem[603] = 32'h00000013;
        imem[604] = 32'h003F2023;
        imem[605] = 32'h00000013;
        imem[606] = 32'h00000013;
        imem[607] = 32'h00000013;
        imem[608] = 32'h20000093;
        imem[609] = 32'h4B400F13;
        imem[610] = 32'h00000013;
        imem[611] = 32'h00000013;
        imem[612] = 32'h00000013;
        imem[613] = 32'h0000A183;
        imem[614] = 32'h00000013;
        imem[615] = 32'h00000013;
        imem[616] = 32'h00000013;
        imem[617] = 32'h003F2023;
        imem[618] = 32'h00000013;
        imem[619] = 32'h00000013;
        imem[620] = 32'h00000013;
        imem[621] = 32'h10000093;
        imem[622] = 32'h00000113;
        imem[623] = 32'h00000013;
        imem[624] = 32'h00000013;
        imem[625] = 32'h00000013;
        imem[626] = 32'h0020A023;
        imem[627] = 32'h00000013;
        imem[628] = 32'h00000013;
        imem[629] = 32'h00000013;
        imem[630] = 32'h10400093;
        imem[631] = 32'h00100113;
        imem[632] = 32'h00000013;
        imem[633] = 32'h00000013;
        imem[634] = 32'h00000013;
        imem[635] = 32'h0020A023;
        imem[636] = 32'h00000013;
        imem[637] = 32'h00000013;
        imem[638] = 32'h00000013;
        imem[639] = 32'h10800093;
        imem[640] = 32'hFFF00113;
        imem[641] = 32'h00000013;
        imem[642] = 32'h00000013;
        imem[643] = 32'h00000013;
        imem[644] = 32'h0020A023;
        imem[645] = 32'h00000013;
        imem[646] = 32'h00000013;
        imem[647] = 32'h00000013;
        imem[648] = 32'h10C00093;
        imem[649] = 32'h7FF00113;
        imem[650] = 32'h00000013;
        imem[651] = 32'h00000013;
        imem[652] = 32'h00000013;
        imem[653] = 32'h0020A023;
        imem[654] = 32'h00000013;
        imem[655] = 32'h00000013;
        imem[656] = 32'h00000013;
        imem[657] = 32'h11000093;
        imem[658] = 32'h80000113;
        imem[659] = 32'h00000013;
        imem[660] = 32'h00000013;
        imem[661] = 32'h00000013;
        imem[662] = 32'h0020A023;
        imem[663] = 32'h00000013;
        imem[664] = 32'h00000013;
        imem[665] = 32'h00000013;
        imem[666] = 32'h11400093;
        imem[667] = 32'h4D200113;
        imem[668] = 32'h00000013;
        imem[669] = 32'h00000013;
        imem[670] = 32'h00000013;
        imem[671] = 32'h0020A023;
        imem[672] = 32'h00000013;
        imem[673] = 32'h00000013;
        imem[674] = 32'h00000013;
        imem[675] = 32'h00A00093;
        imem[676] = 32'h00A00113;
        imem[677] = 32'h4D000F13;
        imem[678] = 32'h2A500E93;
        imem[679] = 32'h00000013;
        imem[680] = 32'h00000013;
        imem[681] = 32'h00000013;
        imem[682] = 32'h00208463;
        imem[683] = 32'h01DF2023;
        imem[684] = 32'h00000013;
        imem[685] = 32'h00000013;
        imem[686] = 32'h00000013;
        imem[687] = 32'h00000013;
        imem[688] = 32'h00000093;
        imem[689] = 32'h00100113;
        imem[690] = 32'h4D400F13;
        imem[691] = 32'h2A500E93;
        imem[692] = 32'h00000013;
        imem[693] = 32'h00000013;
        imem[694] = 32'h00000013;
        imem[695] = 32'h00208463;
        imem[696] = 32'h01DF2023;
        imem[697] = 32'h00000013;
        imem[698] = 32'h00000013;
        imem[699] = 32'h00000013;
        imem[700] = 32'h00000013;
        imem[701] = 32'h80000093;
        imem[702] = 32'h80000113;
        imem[703] = 32'h4D800F13;
        imem[704] = 32'h2A500E93;
        imem[705] = 32'h00000013;
        imem[706] = 32'h00000013;
        imem[707] = 32'h00000013;
        imem[708] = 32'h00208463;
        imem[709] = 32'h01DF2023;
        imem[710] = 32'h00000013;
        imem[711] = 32'h00000013;
        imem[712] = 32'h00000013;
        imem[713] = 32'h00000013;
        imem[714] = 32'h00A00093;
        imem[715] = 32'h01400113;
        imem[716] = 32'h4DC00F13;
        imem[717] = 32'h2A500E93;
        imem[718] = 32'h00000013;
        imem[719] = 32'h00000013;
        imem[720] = 32'h00000013;
        imem[721] = 32'h00209463;
        imem[722] = 32'h01DF2023;
        imem[723] = 32'h00000013;
        imem[724] = 32'h00000013;
        imem[725] = 32'h00000013;
        imem[726] = 32'h00000013;
        imem[727] = 32'hFFF00093;
        imem[728] = 32'hFFF00113;
        imem[729] = 32'h4E000F13;
        imem[730] = 32'h2A500E93;
        imem[731] = 32'h00000013;
        imem[732] = 32'h00000013;
        imem[733] = 32'h00000013;
        imem[734] = 32'h00209463;
        imem[735] = 32'h01DF2023;
        imem[736] = 32'h00000013;
        imem[737] = 32'h00000013;
        imem[738] = 32'h00000013;
        imem[739] = 32'h00000013;
        imem[740] = 32'h7FF00093;
        imem[741] = 32'h80000113;
        imem[742] = 32'h4E400F13;
        imem[743] = 32'h2A500E93;
        imem[744] = 32'h00000013;
        imem[745] = 32'h00000013;
        imem[746] = 32'h00000013;
        imem[747] = 32'h00209463;
        imem[748] = 32'h01DF2023;
        imem[749] = 32'h00000013;
        imem[750] = 32'h00000013;
        imem[751] = 32'h00000013;
        imem[752] = 32'h00000013;
        imem[753] = 32'hFFB00093;
        imem[754] = 32'h00300113;
        imem[755] = 32'h4E800F13;
        imem[756] = 32'h2A500E93;
        imem[757] = 32'h00000013;
        imem[758] = 32'h00000013;
        imem[759] = 32'h00000013;
        imem[760] = 32'h0020C463;
        imem[761] = 32'h01DF2023;
        imem[762] = 32'h00000013;
        imem[763] = 32'h00000013;
        imem[764] = 32'h00000013;
        imem[765] = 32'h00000013;
        imem[766] = 32'h7FF00093;
        imem[767] = 32'h80000113;
        imem[768] = 32'h4EC00F13;
        imem[769] = 32'h2A500E93;
        imem[770] = 32'h00000013;
        imem[771] = 32'h00000013;
        imem[772] = 32'h00000013;
        imem[773] = 32'h0020C463;
        imem[774] = 32'h01DF2023;
        imem[775] = 32'h00000013;
        imem[776] = 32'h00000013;
        imem[777] = 32'h00000013;
        imem[778] = 32'h00000013;
        imem[779] = 32'h80000093;
        imem[780] = 32'h7FF00113;
        imem[781] = 32'h4F000F13;
        imem[782] = 32'h2A500E93;
        imem[783] = 32'h00000013;
        imem[784] = 32'h00000013;
        imem[785] = 32'h00000013;
        imem[786] = 32'h0020C463;
        imem[787] = 32'h01DF2023;
        imem[788] = 32'h00000013;
        imem[789] = 32'h00000013;
        imem[790] = 32'h00000013;
        imem[791] = 32'h00000013;
        imem[792] = 32'h00000093;
        imem[793] = 32'hFFF00113;
        imem[794] = 32'h4F400F13;
        imem[795] = 32'h2A500E93;
        imem[796] = 32'h00000013;
        imem[797] = 32'h00000013;
        imem[798] = 32'h00000013;
        imem[799] = 32'h0020C463;
        imem[800] = 32'h01DF2023;
        imem[801] = 32'h00000013;
        imem[802] = 32'h00000013;
        imem[803] = 32'h00000013;
        imem[804] = 32'h00000013;
        imem[805] = 32'hFFB00093;
        imem[806] = 32'h00300113;
        imem[807] = 32'h4F800F13;
        imem[808] = 32'h2A500E93;
        imem[809] = 32'h00000013;
        imem[810] = 32'h00000013;
        imem[811] = 32'h00000013;
        imem[812] = 32'h0020D463;
        imem[813] = 32'h01DF2023;
        imem[814] = 32'h00000013;
        imem[815] = 32'h00000013;
        imem[816] = 32'h00000013;
        imem[817] = 32'h00000013;
        imem[818] = 32'h00300093;
        imem[819] = 32'hFFB00113;
        imem[820] = 32'h4FC00F13;
        imem[821] = 32'h2A500E93;
        imem[822] = 32'h00000013;
        imem[823] = 32'h00000013;
        imem[824] = 32'h00000013;
        imem[825] = 32'h0020D463;
        imem[826] = 32'h01DF2023;
        imem[827] = 32'h00000013;
        imem[828] = 32'h00000013;
        imem[829] = 32'h00000013;
        imem[830] = 32'h00000013;
        imem[831] = 32'h80000093;
        imem[832] = 32'h80000113;
        imem[833] = 32'h50000F13;
        imem[834] = 32'h2A500E93;
        imem[835] = 32'h00000013;
        imem[836] = 32'h00000013;
        imem[837] = 32'h00000013;
        imem[838] = 32'h0020D463;
        imem[839] = 32'h01DF2023;
        imem[840] = 32'h00000013;
        imem[841] = 32'h00000013;
        imem[842] = 32'h00000013;
        imem[843] = 32'h00000013;
        imem[844] = 32'hFFF00093;
        imem[845] = 32'h7FF00113;
        imem[846] = 32'h50400F13;
        imem[847] = 32'h2A500E93;
        imem[848] = 32'h00000013;
        imem[849] = 32'h00000013;
        imem[850] = 32'h00000013;
        imem[851] = 32'h0020D463;
        imem[852] = 32'h01DF2023;
        imem[853] = 32'h00000013;
        imem[854] = 32'h00000013;
        imem[855] = 32'h00000013;
        imem[856] = 32'h00000013;
        imem[857] = 32'h00000013;
        imem[858] = 32'h00000013;
        imem[859] = 32'h00000013;
        imem[860] = 32'h00000013;
        imem[861] = 32'h00000013;
        imem[862] = 32'h00000013;
        imem[863] = 32'h00000013;
        imem[864] = 32'h00000013;

        $display("\n========================================");
        $display("RISC-V CPU Testbench - deepseek");
        $display("========================================\n");
        #20 rst_n = 1;
        repeat(2675) @(posedge clk);

        total = total + 1;
        if (dmem[256] === 32'h0000001E) begin
            passed = passed + 1;
            $display("\u2713 PASS  add x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  add x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r1]", 32'h0000001E, dmem[256]);
        end
        total = total + 1;
        if (dmem[257] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  add x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  add x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r2]", 32'h00000000, dmem[257]);
        end
        total = total + 1;
        if (dmem[258] === 32'h00000800) begin
            passed = passed + 1;
            $display("\u2713 PASS  add x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  add x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r3]", 32'h00000800, dmem[258]);
        end
        total = total + 1;
        if (dmem[259] === 32'hFFFFF7FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  add x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  add x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r4]", 32'hFFFFF7FF, dmem[259]);
        end
        total = total + 1;
        if (dmem[260] === 32'h00000014) begin
            passed = passed + 1;
            $display("\u2713 PASS  sub x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sub x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r5]", 32'h00000014, dmem[260]);
        end
        total = total + 1;
        if (dmem[261] === 32'hFFFFFFFE) begin
            passed = passed + 1;
            $display("\u2713 PASS  sub x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sub x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r6]", 32'hFFFFFFFE, dmem[261]);
        end
        total = total + 1;
        if (dmem[262] === 32'h00000FFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sub x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sub x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r7]", 32'h00000FFF, dmem[262]);
        end
        total = total + 1;
        if (dmem[263] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sub x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r8]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sub x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r8]", 32'hFFFFFFFF, dmem[263]);
        end
        total = total + 1;
        if (dmem[264] === 32'h0000000F) begin
            passed = passed + 1;
            $display("\u2713 PASS  and x3, x1, x2 [BDD:CPU-LOGICAL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  and x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r1]", 32'h0000000F, dmem[264]);
        end
        total = total + 1;
        if (dmem[265] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  and x3, x1, x2 [BDD:CPU-LOGICAL-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  and x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r2]", 32'h000007FF, dmem[265]);
        end
        total = total + 1;
        if (dmem[266] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  and x3, x1, x2 [BDD:CPU-LOGICAL-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  and x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r3]", 32'h00000000, dmem[266]);
        end
        total = total + 1;
        if (dmem[267] === 32'h000000FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  or x3, x1, x2 [BDD:CPU-LOGICAL-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  or x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r4]", 32'h000000FF, dmem[267]);
        end
        total = total + 1;
        if (dmem[268] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  or x3, x1, x2 [BDD:CPU-LOGICAL-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  or x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r5]", 32'h000007FF, dmem[268]);
        end
        total = total + 1;
        if (dmem[269] === 32'hFFFFF800) begin
            passed = passed + 1;
            $display("\u2713 PASS  or x3, x1, x2 [BDD:CPU-LOGICAL-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  or x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r6]", 32'hFFFFF800, dmem[269]);
        end
        total = total + 1;
        if (dmem[270] === 32'hFFFFF800) begin
            passed = passed + 1;
            $display("\u2713 PASS  xor x3, x1, x2 [BDD:CPU-LOGICAL-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xor x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r7]", 32'hFFFFF800, dmem[270]);
        end
        total = total + 1;
        if (dmem[271] === 32'hFFFFFFFE) begin
            passed = passed + 1;
            $display("\u2713 PASS  xor x3, x1, x2 [BDD:CPU-LOGICAL-001/r8]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xor x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r8]", 32'hFFFFFFFE, dmem[271]);
        end
        total = total + 1;
        if (dmem[272] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  xor x3, x1, x2 [BDD:CPU-LOGICAL-001/r9]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xor x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r9]", 32'hFFFFFFFF, dmem[272]);
        end
        total = total + 1;
        if (dmem[273] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r1]", 32'h00000001, dmem[273]);
        end
        total = total + 1;
        if (dmem[274] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r2]", 32'h00000000, dmem[274]);
        end
        total = total + 1;
        if (dmem[275] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r3]", 32'h00000001, dmem[275]);
        end
        total = total + 1;
        if (dmem[276] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r4]", 32'h00000000, dmem[276]);
        end
        total = total + 1;
        if (dmem[277] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r5]", 32'h00000000, dmem[277]);
        end
        total = total + 1;
        if (dmem[278] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r6]", 32'h00000000, dmem[278]);
        end
        total = total + 1;
        if (dmem[279] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, 0 [BDD:CPU-COMPARE-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, 0 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r7]", 32'h00000001, dmem[279]);
        end
        total = total + 1;
        if (dmem[280] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, -1 [BDD:CPU-COMPARE-001/r8]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, -1 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r8]", 32'h00000001, dmem[280]);
        end
        total = total + 1;
        if (dmem[281] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, 1 [BDD:CPU-COMPARE-001/r9]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, 1 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r9]", 32'h00000001, dmem[281]);
        end
        total = total + 1;
        if (dmem[282] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, 2047 [BDD:CPU-COMPARE-001/r10]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, 2047 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r10]", 32'h00000000, dmem[282]);
        end
        total = total + 1;
        if (dmem[283] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, -2048 [BDD:CPU-COMPARE-001/r11]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, -2048 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r11]", 32'h00000000, dmem[283]);
        end
        total = total + 1;
        if (dmem[284] === 32'h00000096) begin
            passed = passed + 1;
            $display("\u2713 PASS  addi x3, x1, 100 [BDD:CPU-IMMEDIATE-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  addi x3, x1, 100 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r1]", 32'h00000096, dmem[284]);
        end
        total = total + 1;
        if (dmem[285] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  addi x3, x1, -1 [BDD:CPU-IMMEDIATE-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  addi x3, x1, -1 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r2]", 32'hFFFFFFFF, dmem[285]);
        end
        total = total + 1;
        if (dmem[286] === 32'h00000800) begin
            passed = passed + 1;
            $display("\u2713 PASS  addi x3, x1, 2047 [BDD:CPU-IMMEDIATE-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  addi x3, x1, 2047 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r3]", 32'h00000800, dmem[286]);
        end
        total = total + 1;
        if (dmem[287] === 32'hFFFFF000) begin
            passed = passed + 1;
            $display("\u2713 PASS  addi x3, x1, -2048 [BDD:CPU-IMMEDIATE-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  addi x3, x1, -2048 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r4]", 32'hFFFFF000, dmem[287]);
        end
        total = total + 1;
        if (dmem[288] === 32'h0000000F) begin
            passed = passed + 1;
            $display("\u2713 PASS  andi x3, x1, 15 [BDD:CPU-IMMEDIATE-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  andi x3, x1, 15 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r5]", 32'h0000000F, dmem[288]);
        end
        total = total + 1;
        if (dmem[289] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  andi x3, x1, -1 [BDD:CPU-IMMEDIATE-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  andi x3, x1, -1 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r6]", 32'h000007FF, dmem[289]);
        end
        total = total + 1;
        if (dmem[290] === 32'h000000FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  ori x3, x1, 0x0F [BDD:CPU-IMMEDIATE-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  ori x3, x1, 0x0F - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r7]", 32'h000000FF, dmem[290]);
        end
        total = total + 1;
        if (dmem[291] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  ori x3, x1, 2047 [BDD:CPU-IMMEDIATE-001/r8]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  ori x3, x1, 2047 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r8]", 32'h000007FF, dmem[291]);
        end
        total = total + 1;
        if (dmem[292] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  ori x3, x1, -2048 [BDD:CPU-IMMEDIATE-001/r9]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  ori x3, x1, -2048 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r9]", 32'hFFFFFFFF, dmem[292]);
        end
        total = total + 1;
        if (dmem[293] === 32'h000000F0) begin
            passed = passed + 1;
            $display("\u2713 PASS  xori x3, x1, 15 [BDD:CPU-IMMEDIATE-001/r10]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xori x3, x1, 15 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r10]", 32'h000000F0, dmem[293]);
        end
        total = total + 1;
        if (dmem[294] === 32'hFFFFF800) begin
            passed = passed + 1;
            $display("\u2713 PASS  xori x3, x1, -1 [BDD:CPU-IMMEDIATE-001/r11]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xori x3, x1, -1 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r11]", 32'hFFFFF800, dmem[294]);
        end
        total = total + 1;
        if (dmem[295] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  xori x3, x1, -2048 [BDD:CPU-IMMEDIATE-001/r12]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xori x3, x1, -2048 - expected %0d, got %0d [BDD:CPU-IMMEDIATE-001/r12]", 32'h000007FF, dmem[295]);
        end
        total = total + 1;
        if (dmem[296] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r1]", 32'h00000000, dmem[296]);
        end
        total = total + 1;
        if (dmem[297] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r2]", 32'h00000001, dmem[297]);
        end
        total = total + 1;
        if (dmem[298] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r3]", 32'hFFFFFFFF, dmem[298]);
        end
        total = total + 1;
        if (dmem[299] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r4]", 32'h000007FF, dmem[299]);
        end
        total = total + 1;
        if (dmem[300] === 32'hFFFFF800) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r5]", 32'hFFFFF800, dmem[300]);
        end
        total = total + 1;
        if (dmem[301] === 32'h000004D2) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r6]", 32'h000004D2, dmem[301]);
        end
        total = total + 1;
        if (dmem[64] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-002/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-002/r1]", 32'h00000000, dmem[64]);
        end
        total = total + 1;
        if (dmem[65] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-002/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-002/r2]", 32'h00000001, dmem[65]);
        end
        total = total + 1;
        if (dmem[66] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-002/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-002/r3]", 32'hFFFFFFFF, dmem[66]);
        end
        total = total + 1;
        if (dmem[67] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-002/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-002/r4]", 32'h000007FF, dmem[67]);
        end
        total = total + 1;
        if (dmem[68] === 32'hFFFFF800) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-002/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-002/r5]", 32'hFFFFF800, dmem[68]);
        end
        total = total + 1;
        if (dmem[69] === 32'h000004D2) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-002/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-002/r6]", 32'h000004D2, dmem[69]);
        end
        total = total + 1;
        if (dmem[308] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  beq x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  beq x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r1]", 32'h00000000, dmem[308]);
        end
        total = total + 1;
        if (dmem[309] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  beq x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  beq x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r2]", 32'h000002A5, dmem[309]);
        end
        total = total + 1;
        if (dmem[310] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  beq x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  beq x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r3]", 32'h00000000, dmem[310]);
        end
        total = total + 1;
        if (dmem[311] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  bne x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bne x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r4]", 32'h00000000, dmem[311]);
        end
        total = total + 1;
        if (dmem[312] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  bne x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bne x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r5]", 32'h000002A5, dmem[312]);
        end
        total = total + 1;
        if (dmem[313] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  bne x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bne x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r6]", 32'h00000000, dmem[313]);
        end
        total = total + 1;
        if (dmem[314] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  blt x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  blt x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r7]", 32'h00000000, dmem[314]);
        end
        total = total + 1;
        if (dmem[315] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  blt x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r8]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  blt x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r8]", 32'h000002A5, dmem[315]);
        end
        total = total + 1;
        if (dmem[316] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  blt x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r9]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  blt x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r9]", 32'h00000000, dmem[316]);
        end
        total = total + 1;
        if (dmem[317] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  blt x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r10]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  blt x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r10]", 32'h000002A5, dmem[317]);
        end
        total = total + 1;
        if (dmem[318] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  bge x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r11]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bge x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r11]", 32'h000002A5, dmem[318]);
        end
        total = total + 1;
        if (dmem[319] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  bge x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r12]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bge x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r12]", 32'h00000000, dmem[319]);
        end
        total = total + 1;
        if (dmem[320] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  bge x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r13]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bge x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r13]", 32'h00000000, dmem[320]);
        end
        total = total + 1;
        if (dmem[321] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  bge x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r14]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bge x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r14]", 32'h000002A5, dmem[321]);
        end

        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total:  %0d", total);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        if (failed == 0) $display("\n\U0001F389 ALL TESTS PASSED!");
        else $display("\n\u26A0  SOME TESTS FAILED");
        $display("========================================\n");
        $finish;
    end
endmodule