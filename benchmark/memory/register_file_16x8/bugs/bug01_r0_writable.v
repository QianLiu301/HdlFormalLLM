// register_file_16x8 — golden reference design
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
        end else if (we) begin // BUG: R0 writable
            regs[waddr] <= wdata;
        end
    end

    assign rdata_a = regs[raddr_a]; // BUG: R0 not forced to zero
    assign rdata_b = regs[raddr_b]; // BUG: R0 not forced to zero
endmodule
