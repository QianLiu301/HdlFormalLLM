// register_file_16x8 — BUG: internal storage indexed by 3 bits only
module register_file_16x8 (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       we,
    input  wire [3:0] waddr,
    input  wire [7:0] wdata,
    input  wire [3:0] raddr_a,
    input  wire [3:0] raddr_b,
    output wire [7:0] rdata_a,
    output wire [7:0] rdata_b
);
    reg [7:0] regs [0:15];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 16; i = i + 1)
                regs[i] <= 8'h00;
        end else if (we && (waddr != 4'd0)) begin
            regs[waddr[2:0]] <= wdata;  // BUG: addresses 8-15 alias onto 0-7
        end
    end

    assign rdata_a = (raddr_a == 4'd0) ? 8'h00 : regs[raddr_a[2:0]];
    assign rdata_b = (raddr_b == 4'd0) ? 8'h00 : regs[raddr_b[2:0]];
endmodule
