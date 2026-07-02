// sync_fifo_8x8 — golden reference design
module sync_fifo_8x8 (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       wr_en,
    input  wire [7:0] din,
    input  wire       rd_en,
    output reg  [7:0] dout,
    output wire       full,
    output wire       empty
);
    reg [7:0] mem [0:7];
    reg [2:0] wr_ptr, rd_ptr;
    reg [3:0] count;

    wire wr_ok = wr_en && !full;
    wire rd_ok = rd_en && !empty;

    assign full  = (count == 4'd7); // BUG: full one entry early
    assign empty = (count == 4'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 3'd0;
            rd_ptr <= 3'd0;
            count  <= 4'd0;
            dout   <= 8'h00;
        end else begin
            if (wr_ok) begin
                mem[wr_ptr] <= din;
                wr_ptr      <= wr_ptr + 3'd1;
            end
            if (rd_ok) begin
                dout   <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 3'd1;
            end
            case ({wr_ok, rd_ok})
                2'b10: count <= count + 4'd1;
                2'b01: count <= count - 4'd1;
                default: ;
            endcase
        end
    end
endmodule
