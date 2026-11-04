// ==========================================================================
// RISC-V CPU Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: deepseek
// Generated: 2026-11-04 19:54:45
// Oracle: bdd
// Program: 428 instructions, 32 checks
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
        dmem[0] = 32'h00001234;
        dmem[1] = 32'hFFFFFFFF;
        dmem[2] = 32'h000007FF;
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
        imem[14] = 32'h7FF00093;
        imem[15] = 32'h00100113;
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
        imem[28] = 32'h80000093;
        imem[29] = 32'h00100113;
        imem[30] = 32'h40800F13;
        imem[31] = 32'h00000013;
        imem[32] = 32'h00000013;
        imem[33] = 32'h00000013;
        imem[34] = 32'h402081B3;
        imem[35] = 32'h00000013;
        imem[36] = 32'h00000013;
        imem[37] = 32'h00000013;
        imem[38] = 32'h003F2023;
        imem[39] = 32'h00000013;
        imem[40] = 32'h00000013;
        imem[41] = 32'h00000013;
        imem[42] = 32'h00000093;
        imem[43] = 32'hFFF00113;
        imem[44] = 32'h40C00F13;
        imem[45] = 32'h00000013;
        imem[46] = 32'h00000013;
        imem[47] = 32'h00000013;
        imem[48] = 32'h402081B3;
        imem[49] = 32'h00000013;
        imem[50] = 32'h00000013;
        imem[51] = 32'h00000013;
        imem[52] = 32'h003F2023;
        imem[53] = 32'h00000013;
        imem[54] = 32'h00000013;
        imem[55] = 32'h00000013;
        imem[56] = 32'h7FF00093;
        imem[57] = 32'h41000F13;
        imem[58] = 32'h00000013;
        imem[59] = 32'h00000013;
        imem[60] = 32'h00000013;
        imem[61] = 32'h00108193;
        imem[62] = 32'h00000013;
        imem[63] = 32'h00000013;
        imem[64] = 32'h00000013;
        imem[65] = 32'h003F2023;
        imem[66] = 32'h00000013;
        imem[67] = 32'h00000013;
        imem[68] = 32'h00000013;
        imem[69] = 32'h0FF00093;
        imem[70] = 32'h00F00113;
        imem[71] = 32'h41400F13;
        imem[72] = 32'h00000013;
        imem[73] = 32'h00000013;
        imem[74] = 32'h00000013;
        imem[75] = 32'h0020F1B3;
        imem[76] = 32'h00000013;
        imem[77] = 32'h00000013;
        imem[78] = 32'h00000013;
        imem[79] = 32'h003F2023;
        imem[80] = 32'h00000013;
        imem[81] = 32'h00000013;
        imem[82] = 32'h00000013;
        imem[83] = 32'h0F000093;
        imem[84] = 32'h00F00113;
        imem[85] = 32'h41800F13;
        imem[86] = 32'h00000013;
        imem[87] = 32'h00000013;
        imem[88] = 32'h00000013;
        imem[89] = 32'h0020E1B3;
        imem[90] = 32'h00000013;
        imem[91] = 32'h00000013;
        imem[92] = 32'h00000013;
        imem[93] = 32'h003F2023;
        imem[94] = 32'h00000013;
        imem[95] = 32'h00000013;
        imem[96] = 32'h00000013;
        imem[97] = 32'h0FF00093;
        imem[98] = 32'h00F00113;
        imem[99] = 32'h41C00F13;
        imem[100] = 32'h00000013;
        imem[101] = 32'h00000013;
        imem[102] = 32'h00000013;
        imem[103] = 32'h0020C1B3;
        imem[104] = 32'h00000013;
        imem[105] = 32'h00000013;
        imem[106] = 32'h00000013;
        imem[107] = 32'h003F2023;
        imem[108] = 32'h00000013;
        imem[109] = 32'h00000013;
        imem[110] = 32'h00000013;
        imem[111] = 32'h0F000093;
        imem[112] = 32'h42000F13;
        imem[113] = 32'h00000013;
        imem[114] = 32'h00000013;
        imem[115] = 32'h00000013;
        imem[116] = 32'h00F0F193;
        imem[117] = 32'h00000013;
        imem[118] = 32'h00000013;
        imem[119] = 32'h00000013;
        imem[120] = 32'h003F2023;
        imem[121] = 32'h00000013;
        imem[122] = 32'h00000013;
        imem[123] = 32'h00000013;
        imem[124] = 32'h0F000093;
        imem[125] = 32'h42400F13;
        imem[126] = 32'h00000013;
        imem[127] = 32'h00000013;
        imem[128] = 32'h00000013;
        imem[129] = 32'h00F0E193;
        imem[130] = 32'h00000013;
        imem[131] = 32'h00000013;
        imem[132] = 32'h00000013;
        imem[133] = 32'h003F2023;
        imem[134] = 32'h00000013;
        imem[135] = 32'h00000013;
        imem[136] = 32'h00000013;
        imem[137] = 32'h00100093;
        imem[138] = 32'h42800F13;
        imem[139] = 32'h00000013;
        imem[140] = 32'h00000013;
        imem[141] = 32'h00000013;
        imem[142] = 32'hFFF0C193;
        imem[143] = 32'h00000013;
        imem[144] = 32'h00000013;
        imem[145] = 32'h00000013;
        imem[146] = 32'h003F2023;
        imem[147] = 32'h00000013;
        imem[148] = 32'h00000013;
        imem[149] = 32'h00000013;
        imem[150] = 32'h80000093;
        imem[151] = 32'h7FF00113;
        imem[152] = 32'h42C00F13;
        imem[153] = 32'h00000013;
        imem[154] = 32'h00000013;
        imem[155] = 32'h00000013;
        imem[156] = 32'h0020F1B3;
        imem[157] = 32'h00000013;
        imem[158] = 32'h00000013;
        imem[159] = 32'h00000013;
        imem[160] = 32'h003F2023;
        imem[161] = 32'h00000013;
        imem[162] = 32'h00000013;
        imem[163] = 32'h00000013;
        imem[164] = 32'hFFF00093;
        imem[165] = 32'h00100113;
        imem[166] = 32'h43000F13;
        imem[167] = 32'h00000013;
        imem[168] = 32'h00000013;
        imem[169] = 32'h00000013;
        imem[170] = 32'h0020A1B3;
        imem[171] = 32'h00000013;
        imem[172] = 32'h00000013;
        imem[173] = 32'h00000013;
        imem[174] = 32'h003F2023;
        imem[175] = 32'h00000013;
        imem[176] = 32'h00000013;
        imem[177] = 32'h00000013;
        imem[178] = 32'h00500093;
        imem[179] = 32'h00300113;
        imem[180] = 32'h43400F13;
        imem[181] = 32'h00000013;
        imem[182] = 32'h00000013;
        imem[183] = 32'h00000013;
        imem[184] = 32'h0020A1B3;
        imem[185] = 32'h00000013;
        imem[186] = 32'h00000013;
        imem[187] = 32'h00000013;
        imem[188] = 32'h003F2023;
        imem[189] = 32'h00000013;
        imem[190] = 32'h00000013;
        imem[191] = 32'h00000013;
        imem[192] = 32'hFFF00093;
        imem[193] = 32'h43800F13;
        imem[194] = 32'h00000013;
        imem[195] = 32'h00000013;
        imem[196] = 32'h00000013;
        imem[197] = 32'h0000A193;
        imem[198] = 32'h00000013;
        imem[199] = 32'h00000013;
        imem[200] = 32'h00000013;
        imem[201] = 32'h003F2023;
        imem[202] = 32'h00000013;
        imem[203] = 32'h00000013;
        imem[204] = 32'h00000013;
        imem[205] = 32'h80000093;
        imem[206] = 32'h7FF00113;
        imem[207] = 32'h43C00F13;
        imem[208] = 32'h00000013;
        imem[209] = 32'h00000013;
        imem[210] = 32'h00000013;
        imem[211] = 32'h0020A1B3;
        imem[212] = 32'h00000013;
        imem[213] = 32'h00000013;
        imem[214] = 32'h00000013;
        imem[215] = 32'h003F2023;
        imem[216] = 32'h00000013;
        imem[217] = 32'h00000013;
        imem[218] = 32'h00000013;
        imem[219] = 32'h7FF00093;
        imem[220] = 32'h80000113;
        imem[221] = 32'h44000F13;
        imem[222] = 32'h00000013;
        imem[223] = 32'h00000013;
        imem[224] = 32'h00000013;
        imem[225] = 32'h0020A1B3;
        imem[226] = 32'h00000013;
        imem[227] = 32'h00000013;
        imem[228] = 32'h00000013;
        imem[229] = 32'h003F2023;
        imem[230] = 32'h00000013;
        imem[231] = 32'h00000013;
        imem[232] = 32'h00000013;
        imem[233] = 32'h7FF00093;
        imem[234] = 32'h44400F13;
        imem[235] = 32'h00000013;
        imem[236] = 32'h00000013;
        imem[237] = 32'h00000013;
        imem[238] = 32'h7FF0A193;
        imem[239] = 32'h00000013;
        imem[240] = 32'h00000013;
        imem[241] = 32'h00000013;
        imem[242] = 32'h003F2023;
        imem[243] = 32'h00000013;
        imem[244] = 32'h00000013;
        imem[245] = 32'h00000013;
        imem[246] = 32'h80000093;
        imem[247] = 32'h44800F13;
        imem[248] = 32'h00000013;
        imem[249] = 32'h00000013;
        imem[250] = 32'h00000013;
        imem[251] = 32'h8000A193;
        imem[252] = 32'h00000013;
        imem[253] = 32'h00000013;
        imem[254] = 32'h00000013;
        imem[255] = 32'h003F2023;
        imem[256] = 32'h00000013;
        imem[257] = 32'h00000013;
        imem[258] = 32'h00000013;
        imem[259] = 32'h00000093;
        imem[260] = 32'h44C00F13;
        imem[261] = 32'h00000013;
        imem[262] = 32'h00000013;
        imem[263] = 32'h00000013;
        imem[264] = 32'h0000A183;
        imem[265] = 32'h00000013;
        imem[266] = 32'h00000013;
        imem[267] = 32'h00000013;
        imem[268] = 32'h003F2023;
        imem[269] = 32'h00000013;
        imem[270] = 32'h00000013;
        imem[271] = 32'h00000013;
        imem[272] = 32'h00400093;
        imem[273] = 32'h45000F13;
        imem[274] = 32'h00000013;
        imem[275] = 32'h00000013;
        imem[276] = 32'h00000013;
        imem[277] = 32'h0000A183;
        imem[278] = 32'h00000013;
        imem[279] = 32'h00000013;
        imem[280] = 32'h00000013;
        imem[281] = 32'h003F2023;
        imem[282] = 32'h00000013;
        imem[283] = 32'h00000013;
        imem[284] = 32'h00000013;
        imem[285] = 32'h00800093;
        imem[286] = 32'h45400F13;
        imem[287] = 32'h00000013;
        imem[288] = 32'h00000013;
        imem[289] = 32'h00000013;
        imem[290] = 32'h0000A183;
        imem[291] = 32'h00000013;
        imem[292] = 32'h00000013;
        imem[293] = 32'h00000013;
        imem[294] = 32'h003F2023;
        imem[295] = 32'h00000013;
        imem[296] = 32'h00000013;
        imem[297] = 32'h00000013;
        imem[298] = 32'h10000093;
        imem[299] = 32'h7FF00113;
        imem[300] = 32'h00000013;
        imem[301] = 32'h00000013;
        imem[302] = 32'h00000013;
        imem[303] = 32'h0020A023;
        imem[304] = 32'h00000013;
        imem[305] = 32'h00000013;
        imem[306] = 32'h00000013;
        imem[307] = 32'h20000093;
        imem[308] = 32'hFFF00113;
        imem[309] = 32'h00000013;
        imem[310] = 32'h00000013;
        imem[311] = 32'h00000013;
        imem[312] = 32'h0020A023;
        imem[313] = 32'h00000013;
        imem[314] = 32'h00000013;
        imem[315] = 32'h00000013;
        imem[316] = 32'h00A00093;
        imem[317] = 32'h00A00113;
        imem[318] = 32'h46000F13;
        imem[319] = 32'h2A500E93;
        imem[320] = 32'h00000013;
        imem[321] = 32'h00000013;
        imem[322] = 32'h00000013;
        imem[323] = 32'h00208463;
        imem[324] = 32'h01DF2023;
        imem[325] = 32'h00000013;
        imem[326] = 32'h00000013;
        imem[327] = 32'h00000013;
        imem[328] = 32'h00000013;
        imem[329] = 32'h00A00093;
        imem[330] = 32'h01400113;
        imem[331] = 32'h46400F13;
        imem[332] = 32'h2A500E93;
        imem[333] = 32'h00000013;
        imem[334] = 32'h00000013;
        imem[335] = 32'h00000013;
        imem[336] = 32'h00208463;
        imem[337] = 32'h01DF2023;
        imem[338] = 32'h00000013;
        imem[339] = 32'h00000013;
        imem[340] = 32'h00000013;
        imem[341] = 32'h00000013;
        imem[342] = 32'h00A00093;
        imem[343] = 32'h01400113;
        imem[344] = 32'h46800F13;
        imem[345] = 32'h2A500E93;
        imem[346] = 32'h00000013;
        imem[347] = 32'h00000013;
        imem[348] = 32'h00000013;
        imem[349] = 32'h00209463;
        imem[350] = 32'h01DF2023;
        imem[351] = 32'h00000013;
        imem[352] = 32'h00000013;
        imem[353] = 32'h00000013;
        imem[354] = 32'h00000013;
        imem[355] = 32'h00A00093;
        imem[356] = 32'h00A00113;
        imem[357] = 32'h46C00F13;
        imem[358] = 32'h2A500E93;
        imem[359] = 32'h00000013;
        imem[360] = 32'h00000013;
        imem[361] = 32'h00000013;
        imem[362] = 32'h00209463;
        imem[363] = 32'h01DF2023;
        imem[364] = 32'h00000013;
        imem[365] = 32'h00000013;
        imem[366] = 32'h00000013;
        imem[367] = 32'h00000013;
        imem[368] = 32'h80000093;
        imem[369] = 32'h7FF00113;
        imem[370] = 32'h47000F13;
        imem[371] = 32'h2A500E93;
        imem[372] = 32'h00000013;
        imem[373] = 32'h00000013;
        imem[374] = 32'h00000013;
        imem[375] = 32'h0020C463;
        imem[376] = 32'h01DF2023;
        imem[377] = 32'h00000013;
        imem[378] = 32'h00000013;
        imem[379] = 32'h00000013;
        imem[380] = 32'h00000013;
        imem[381] = 32'h7FF00093;
        imem[382] = 32'h80000113;
        imem[383] = 32'h47400F13;
        imem[384] = 32'h2A500E93;
        imem[385] = 32'h00000013;
        imem[386] = 32'h00000013;
        imem[387] = 32'h00000013;
        imem[388] = 32'h0020C463;
        imem[389] = 32'h01DF2023;
        imem[390] = 32'h00000013;
        imem[391] = 32'h00000013;
        imem[392] = 32'h00000013;
        imem[393] = 32'h00000013;
        imem[394] = 32'hFFF00093;
        imem[395] = 32'h00100113;
        imem[396] = 32'h47800F13;
        imem[397] = 32'h2A500E93;
        imem[398] = 32'h00000013;
        imem[399] = 32'h00000013;
        imem[400] = 32'h00000013;
        imem[401] = 32'h0020D463;
        imem[402] = 32'h01DF2023;
        imem[403] = 32'h00000013;
        imem[404] = 32'h00000013;
        imem[405] = 32'h00000013;
        imem[406] = 32'h00000013;
        imem[407] = 32'h00100093;
        imem[408] = 32'hFFF00113;
        imem[409] = 32'h47C00F13;
        imem[410] = 32'h2A500E93;
        imem[411] = 32'h00000013;
        imem[412] = 32'h00000013;
        imem[413] = 32'h00000013;
        imem[414] = 32'h0020D463;
        imem[415] = 32'h01DF2023;
        imem[416] = 32'h00000013;
        imem[417] = 32'h00000013;
        imem[418] = 32'h00000013;
        imem[419] = 32'h00000013;
        imem[420] = 32'h00000013;
        imem[421] = 32'h00000013;
        imem[422] = 32'h00000013;
        imem[423] = 32'h00000013;
        imem[424] = 32'h00000013;
        imem[425] = 32'h00000013;
        imem[426] = 32'h00000013;
        imem[427] = 32'h00000013;

        $display("\n========================================");
        $display("RISC-V CPU Testbench - deepseek");
        $display("========================================\n");
        #20 rst_n = 1;
        repeat(1364) @(posedge clk);

        total = total + 1;
        if (dmem[256] === 32'h0000001E) begin
            passed = passed + 1;
            $display("\u2713 PASS  add x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  add x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r1]", 32'h0000001E, dmem[256]);
        end
        total = total + 1;
        if (dmem[257] === 32'h00000800) begin
            passed = passed + 1;
            $display("\u2713 PASS  add x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  add x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r2]", 32'h00000800, dmem[257]);
        end
        total = total + 1;
        if (dmem[258] === 32'hFFFFF7FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sub x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sub x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r3]", 32'hFFFFF7FF, dmem[258]);
        end
        total = total + 1;
        if (dmem[259] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  sub x3, x1, x2 [BDD:CPU-ARITHMETIC-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sub x3, x1, x2 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r4]", 32'h00000001, dmem[259]);
        end
        total = total + 1;
        if (dmem[260] === 32'h00000800) begin
            passed = passed + 1;
            $display("\u2713 PASS  addi x3, x1, 1 [BDD:CPU-ARITHMETIC-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  addi x3, x1, 1 - expected %0d, got %0d [BDD:CPU-ARITHMETIC-001/r5]", 32'h00000800, dmem[260]);
        end
        total = total + 1;
        if (dmem[261] === 32'h0000000F) begin
            passed = passed + 1;
            $display("\u2713 PASS  and x3, x1, x2 [BDD:CPU-LOGICAL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  and x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r1]", 32'h0000000F, dmem[261]);
        end
        total = total + 1;
        if (dmem[262] === 32'h000000FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  or x3, x1, x2 [BDD:CPU-LOGICAL-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  or x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r2]", 32'h000000FF, dmem[262]);
        end
        total = total + 1;
        if (dmem[263] === 32'h000000F0) begin
            passed = passed + 1;
            $display("\u2713 PASS  xor x3, x1, x2 [BDD:CPU-LOGICAL-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xor x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r3]", 32'h000000F0, dmem[263]);
        end
        total = total + 1;
        if (dmem[264] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  andi x3, x1, 15 [BDD:CPU-LOGICAL-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  andi x3, x1, 15 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r4]", 32'h00000000, dmem[264]);
        end
        total = total + 1;
        if (dmem[265] === 32'h000000FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  ori x3, x1, 15 [BDD:CPU-LOGICAL-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  ori x3, x1, 15 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r5]", 32'h000000FF, dmem[265]);
        end
        total = total + 1;
        if (dmem[266] === 32'hFFFFFFFE) begin
            passed = passed + 1;
            $display("\u2713 PASS  xori x3, x1, -1 [BDD:CPU-LOGICAL-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  xori x3, x1, -1 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r6]", 32'hFFFFFFFE, dmem[266]);
        end
        total = total + 1;
        if (dmem[267] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  and x3, x1, x2 [BDD:CPU-LOGICAL-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  and x3, x1, x2 - expected %0d, got %0d [BDD:CPU-LOGICAL-001/r7]", 32'h00000000, dmem[267]);
        end
        total = total + 1;
        if (dmem[268] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r1]", 32'h00000001, dmem[268]);
        end
        total = total + 1;
        if (dmem[269] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r2]", 32'h00000000, dmem[269]);
        end
        total = total + 1;
        if (dmem[270] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, 0 [BDD:CPU-COMPARE-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, 0 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r3]", 32'h00000001, dmem[270]);
        end
        total = total + 1;
        if (dmem[271] === 32'h00000001) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r4]", 32'h00000001, dmem[271]);
        end
        total = total + 1;
        if (dmem[272] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slt x3, x1, x2 [BDD:CPU-COMPARE-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slt x3, x1, x2 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r5]", 32'h00000000, dmem[272]);
        end
        total = total + 1;
        if (dmem[273] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, 2047 [BDD:CPU-COMPARE-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, 2047 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r6]", 32'h00000000, dmem[273]);
        end
        total = total + 1;
        if (dmem[274] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  slti x3, x1, -2048 [BDD:CPU-COMPARE-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  slti x3, x1, -2048 - expected %0d, got %0d [BDD:CPU-COMPARE-001/r7]", 32'h00000000, dmem[274]);
        end
        total = total + 1;
        if (dmem[275] === 32'h00001234) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r1]", 32'h00001234, dmem[275]);
        end
        total = total + 1;
        if (dmem[276] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r2]", 32'hFFFFFFFF, dmem[276]);
        end
        total = total + 1;
        if (dmem[277] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  lw x3, 0(x1) [BDD:CPU-MEMORY-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  lw x3, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r3]", 32'h000007FF, dmem[277]);
        end
        total = total + 1;
        if (dmem[64] === 32'h000007FF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r4]", 32'h000007FF, dmem[64]);
        end
        total = total + 1;
        if (dmem[128] === 32'hFFFFFFFF) begin
            passed = passed + 1;
            $display("\u2713 PASS  sw x2, 0(x1) [BDD:CPU-MEMORY-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  sw x2, 0(x1) - expected %0d, got %0d [BDD:CPU-MEMORY-001/r5]", 32'hFFFFFFFF, dmem[128]);
        end
        total = total + 1;
        if (dmem[280] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  beq x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r1]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  beq x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r1]", 32'h00000000, dmem[280]);
        end
        total = total + 1;
        if (dmem[281] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  beq x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r2]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  beq x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r2]", 32'h000002A5, dmem[281]);
        end
        total = total + 1;
        if (dmem[282] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  bne x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r3]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bne x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r3]", 32'h00000000, dmem[282]);
        end
        total = total + 1;
        if (dmem[283] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  bne x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r4]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bne x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r4]", 32'h000002A5, dmem[283]);
        end
        total = total + 1;
        if (dmem[284] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  blt x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r5]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  blt x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r5]", 32'h00000000, dmem[284]);
        end
        total = total + 1;
        if (dmem[285] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  blt x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r6]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  blt x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r6]", 32'h000002A5, dmem[285]);
        end
        total = total + 1;
        if (dmem[286] === 32'h000002A5) begin
            passed = passed + 1;
            $display("\u2713 PASS  bge x1, x2, 8 (not taken) [BDD:CPU-BRANCH-001/r7]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bge x1, x2, 8 (not taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r7]", 32'h000002A5, dmem[286]);
        end
        total = total + 1;
        if (dmem[287] === 32'h00000000) begin
            passed = passed + 1;
            $display("\u2713 PASS  bge x1, x2, 8 (taken) [BDD:CPU-BRANCH-001/r8]");
        end else begin
            failed = failed + 1;
            $display("\u2717 FAIL  bge x1, x2, 8 (taken) - expected %0d, got %0d [BDD:CPU-BRANCH-001/r8]", 32'h00000000, dmem[287]);
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